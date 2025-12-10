#!/usr/bin/env python3
"""
Music Organizer Script
Scans a music folder, identifies songs/albums/artists using Gemini API,
and reorganizes them into Artist/Album/Song folder structure in-place.
Properly handles compilation albums with Various Artists.

Features:
- Gemini AI for missing metadata identification
- Compilation album detection (Various Artists)
- Safe copy-verify-delete file operations
- JSON change log for auditing
- Undo script generation
- Resume support (skip already organized)
- Rate limiting for API calls
- Response caching to reduce API costs
- Graceful interrupt handling
"""

import os
import sys
import argparse
import shutil
import json
import re
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

from dotenv import load_dotenv
import google.generativeai as genai
from mutagen import File as MutagenFile

# Load environment variables from .env file
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Can override via .env
SUPPORTED_FORMATS = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'}
API_RATE_LIMIT_DELAY = 0.1  # seconds between batch API calls (2000 RPM = plenty of headroom)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5  # seconds, will exponentially increase
SKIP_DIRECTORIES = {'.git', '.svn', '__pycache__', 'node_modules'}
BATCH_SIZE = 10  # Number of files to identify per API call

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Global logger for interrupt handling
_current_logger: Optional['ChangeLogger'] = None
_current_log_file: Optional[Path] = None
_current_undo_script: Optional[Path] = None


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully - save logs before exit."""
    print("\n\n⚠️  Interrupted! Saving progress...")
    if _current_logger and _current_log_file:
        try:
            _current_logger.save()
            print(f"📝 Partial log saved: {_current_log_file.name}")
            if _current_undo_script:
                _current_logger.generate_undo_script(_current_undo_script)
                print(f"↩️  Partial undo script saved: {_current_undo_script.name}")
        except Exception as e:
            print(f"❌ Could not save logs: {e}")
    sys.exit(1)


# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class GeminiCache:
    """Simple cache for Gemini API responses to avoid duplicate calls."""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self):
        """Load cache from file if exists."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
    
    def _save(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        """Cache a response."""
        self.cache[key] = value
        self._save()


class ChangeLogger:
    """Logs all file changes to JSON for auditing and undo."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.changes: List[Dict[str, Any]] = []
        self.start_time = datetime.now().isoformat()
    
    def log_move(self, source: Path, dest: Path, metadata: Dict[str, Any]):
        """Log a file move operation."""
        self.changes.append({
            'action': 'move',
            'source': str(source),
            'destination': str(dest),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_skip(self, file_path: Path, reason: str):
        """Log a skipped file."""
        self.changes.append({
            'action': 'skip',
            'file': str(file_path),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_error(self, file_path: Path, error: str):
        """Log an error."""
        self.changes.append({
            'action': 'error',
            'file': str(file_path),
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def save(self):
        """Save log to file."""
        log_data = {
            'start_time': self.start_time,
            'end_time': datetime.now().isoformat(),
            'total_changes': len([c for c in self.changes if c['action'] == 'move']),
            'changes': self.changes
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def generate_undo_script(self, script_path: Path):
        """Generate a shell script to undo all moves."""
        moves = [c for c in self.changes if c['action'] == 'move']
        if not moves:
            return
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write("# Undo script generated by Music Organizer\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Total files to restore: {len(moves)}\n\n")
            f.write("set -e\n\n")
            
            for move in reversed(moves):
                source = move['source'].replace("'", "'\\''")
                dest = move['destination'].replace("'", "'\\''")
                # Create parent directory if needed
                parent = str(Path(move['source']).parent).replace("'", "'\\''")
                f.write(f"mkdir -p '{parent}'\n")
                f.write(f"mv '{dest}' '{source}'\n")
            
            f.write("\necho 'Undo complete!'\n")
        
        # Make executable
        os.chmod(script_path, 0o755)


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename/folder name."""
    if not name:
        return "Unknown"
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '')
    # Remove leading/trailing spaces and dots
    name = name.strip(' .')
    # Limit length
    return name[:100] if name else "Unknown"


def extract_metadata(file_path: Path) -> Dict[str, Optional[str]]:
    """Extract metadata from audio file using mutagen."""
    metadata: Dict[str, Optional[str]] = {
        'artist': None,
        'album_artist': None,
        'album': None,
        'title': None,
        'track_number': None
    }
    
    try:
        audio = MutagenFile(file_path, easy=True)
        if audio is None:
            return metadata
        
        # Try to get metadata
        if 'artist' in audio:
            metadata['artist'] = audio['artist'][0] if audio['artist'] else None
        if 'albumartist' in audio:
            metadata['album_artist'] = audio['albumartist'][0] if audio['albumartist'] else None
        if 'album' in audio:
            metadata['album'] = audio['album'][0] if audio['album'] else None
        if 'title' in audio:
            metadata['title'] = audio['title'][0] if audio['title'] else None
        if 'tracknumber' in audio:
            track = audio['tracknumber'][0] if audio['tracknumber'] else None
            if track:
                # Handle "1/12" format
                metadata['track_number'] = track.split('/')[0].zfill(2)
    except Exception as e:
        print(f"  Warning: Could not read metadata from {file_path.name}: {e}")
    
    return metadata


def identify_batch_with_gemini(files_data: List[Dict[str, Any]], 
                                cache: Optional[GeminiCache] = None,
                                quiet: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Batch identify multiple files with a single Gemini API call.
    Returns dict mapping filename to identified metadata.
    """
    results: Dict[str, Dict[str, Any]] = {}
    uncached_files: List[Dict[str, Any]] = []
    
    # Check cache first for each file
    for file_data in files_data:
        filename = file_data['filename']
        metadata = file_data['metadata']
        cache_key = f"{filename}|{metadata.get('artist')}|{metadata.get('album')}"
        
        if cache:
            cached = cache.get(cache_key)
            if cached:
                results[filename] = cached
                continue
        
        uncached_files.append(file_data)
    
    if not uncached_files:
        return results
    
    # Build batch context
    files_context = []
    for i, file_data in enumerate(uncached_files, 1):
        filename = file_data['filename']
        metadata = file_data['metadata']
        
        parts = [f"File {i}: {filename}"]
        if metadata.get('artist'):
            parts.append(f"  Artist: {metadata['artist']}")
        if metadata.get('album'):
            parts.append(f"  Album: {metadata['album']}")
        if metadata.get('title'):
            parts.append(f"  Title: {metadata['title']}")
        files_context.append("\n".join(parts))
    
    batch_context = "\n\n".join(files_context)
    
    prompt = f"""Analyze these {len(uncached_files)} music files and identify the track artist, album artist, album, and song title for each.

{batch_context}

IMPORTANT RULES:
1. For COMPILATION albums (soundtracks, "Various Artists", etc.): album_artist = "Various Artists"
2. For regular albums: album_artist = artist
3. ALWAYS try to identify the correct album name. You have extensive music knowledge - use it!
   - If you recognize the song and artist, provide the ACTUAL album it was released on
   - For example: "It's My Life" by Bon Jovi is from the album "Crush"
   - Only use "Singles" if the song was never released on a studio album
   - NEVER return "Unknown" for album if you can reasonably guess it

Respond with a JSON array of objects, one per file, in order:
[
  {{"file": 1, "artist": "...", "album_artist": "...", "album": "...", "title": "...", "is_compilation": false}},
  {{"file": 2, "artist": "...", "album_artist": "...", "album": "...", "title": "...", "is_compilation": false}}
]

No markdown, just the JSON array.
"""

    response_text = ""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Rate limiting after successful call
            time.sleep(API_RATE_LIMIT_DELAY)
            
            # Clean up response
            if response_text.startswith('```'):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
                response_text = re.sub(r'\s*```$', '', response_text)
            
            batch_results = json.loads(response_text)
            
            # Map results back to filenames
            for i, file_data in enumerate(uncached_files):
                filename = file_data['filename']
                metadata = file_data['metadata']
                
                if i < len(batch_results):
                    result = batch_results[i]
                    identified: Dict[str, Any] = {
                        'artist': sanitize_filename(result.get('artist', 'Unknown Artist')),
                        'album_artist': sanitize_filename(result.get('album_artist', result.get('artist', 'Unknown Artist'))),
                        'album': sanitize_filename(result.get('album', 'Unknown Album')),
                        'title': sanitize_filename(result.get('title', 'Unknown Title')),
                        'is_compilation': result.get('is_compilation', False)
                    }
                else:
                    # Fallback if response is incomplete
                    artist = metadata.get('artist') or 'Unknown Artist'
                    identified = {
                        'artist': artist,
                        'album_artist': metadata.get('album_artist') or artist,
                        'album': metadata.get('album') or 'Unknown Album',
                        'title': metadata.get('title') or Path(filename).stem,
                        'is_compilation': False
                    }
                
                results[filename] = identified
                
                # Cache result
                if cache:
                    cache_key = f"{filename}|{metadata.get('artist')}|{metadata.get('album')}"
                    cache.set(cache_key, identified)
            
            return results
            
        except json.JSONDecodeError as e:
            if not quiet:
                print(f"  Warning: Could not parse Gemini response: {e}")
            last_error = e
            break  # Don't retry parse errors
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check for rate limit (429)
            if '429' in error_str or 'quota' in error_str.lower():
                retry_delay = RETRY_BASE_DELAY * (2 ** attempt)
                if not quiet:
                    print(f"  ⏳ Rate limited. Waiting {retry_delay}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(retry_delay)
                continue
            else:
                if not quiet:
                    print(f"  Warning: Gemini API error: {e}")
                break  # Don't retry other errors
    
    # Fallback for all files in batch after all retries failed
    if not quiet and last_error:
        print(f"  Using fallback metadata after API failure")
        
    for file_data in uncached_files:
        filename = file_data['filename']
        metadata = file_data['metadata']
        artist = metadata.get('artist') or 'Unknown Artist'
        results[filename] = {
            'artist': artist,
            'album_artist': metadata.get('album_artist') or artist,
            'album': metadata.get('album') or 'Unknown Album',
            'title': metadata.get('title') or Path(filename).stem,
            'is_compilation': False
        }
    
    return results


def is_compilation(metadata: Dict[str, Optional[str]], identified: Dict[str, Any]) -> bool:
    """Detect if this is a compilation album."""
    # Check if Gemini identified it as compilation
    if identified.get('is_compilation'):
        return True
    
    # Check album_artist tag
    album_artist = metadata.get('album_artist') or identified.get('album_artist', '')
    if album_artist and album_artist.lower() in ['various artists', 'various', 'va', 'soundtrack', 'ost']:
        return True
    
    # Check if album_artist differs from track artist
    if metadata.get('album_artist') and metadata.get('artist'):
        if metadata['album_artist'].lower() != metadata['artist'].lower():
            if 'various' in metadata['album_artist'].lower():
                return True
    
    return False


def scan_music_folder(folder_path: Path) -> List[Path]:
    """Recursively scan folder for audio files, skipping hidden/system directories."""
    audio_files: List[Path] = []
    
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden and system directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in SKIP_DIRECTORIES]
        
        for file in files:
            # Skip hidden files
            if file.startswith('.'):
                continue
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_FORMATS:
                audio_files.append(file_path)
    
    return audio_files


def is_already_organized(file_path: Path, folder_path: Path) -> bool:
    """Check if file is already in Artist/Album structure with valid naming."""
    try:
        relative = file_path.relative_to(folder_path)
        parts = relative.parts
        
        # Must be in Artist/Album/file.mp3 structure (depth of 3)
        if len(parts) != 3:
            return False
        
        artist_folder, album_folder, filename = parts
        
        # Check that folders are not empty/placeholder names
        if artist_folder in ('', '.', '..', 'Unknown', 'Unknown Artist'):
            return False
        if album_folder in ('', '.', '..', 'Unknown', 'Unknown Album'):
            return False
        
        # Check filename follows pattern: "NN - Title.ext" or "NN - Artist - Title.ext"
        if not re.match(r'^\d{2}\s*-\s*.+\..+$', filename):
            return False
        
        return True
    except Exception:
        return False


def process_file(file_path: Path, folder_path: Path, metadata: Dict[str, Optional[str]],
                 identified: Dict[str, Any], dry_run: bool, verbose: bool,
                 quiet: bool, logger: ChangeLogger) -> Tuple[str, Optional[Path]]:
    """
    Process a single file for organization.
    Returns: (status, dest_path) where status is 'success', 'skip', or 'error'
    """
    # Detect if this is a compilation
    is_comp = is_compilation(metadata, identified)
    
    # Use album_artist for folder (handles compilations properly)
    if is_comp:
        folder_artist = "Various Artists"
    else:
        folder_artist = identified.get('album_artist') or identified['artist']
    
    # Build destination path
    artist_folder = folder_path / folder_artist
    album_folder = artist_folder / identified['album']
    
    # Build new filename
    track_num = metadata.get('track_number') or "00"
    
    if is_comp:
        # For compilations: include track artist in filename
        new_filename = f"{track_num} - {identified['artist']} - {identified['title']}{file_path.suffix.lower()}"
    else:
        # For regular albums: just track number and title
        new_filename = f"{track_num} - {identified['title']}{file_path.suffix.lower()}"
    
    dest_path = album_folder / new_filename
    
    if verbose:
        print(f"  → Artist: {identified['artist']}")
        print(f"  → Album Artist: {folder_artist}")
        print(f"  → Album: {identified['album']}")
        print(f"  → Title: {identified['title']}")
        print(f"  → Compilation: {'Yes' if is_comp else 'No'}")
        print(f"  → Destination: {dest_path.relative_to(folder_path)}")
    
    # Skip if already in correct location (compare normalized string paths)
    source_normalized = str(file_path.resolve()).lower()
    dest_normalized = str((album_folder / new_filename).resolve()).lower()
    
    if source_normalized == dest_normalized:
        if verbose:
            print(f"  ⏭️  Already organized")
        logger.log_skip(file_path, "Already at destination")
        return 'skip', None
    
    # Handle duplicates
    if dest_path.exists():
        counter = 1
        stem = dest_path.stem
        while dest_path.exists():
            new_name = f"{stem} ({counter}){dest_path.suffix}"
            dest_path = dest_path.parent / new_name
            counter += 1
    
    if not dry_run:
        # Create directories
        album_folder.mkdir(parents=True, exist_ok=True)
        
        # Direct move (instant on same filesystem via os.rename)
        try:
            shutil.move(str(file_path), str(dest_path))
            if verbose:
                print(f"  ✅ Moved to: {dest_path.relative_to(folder_path)}")
            logger.log_move(file_path, dest_path, identified)
            return 'success', dest_path
        except Exception as e:
            if not quiet:
                print(f"  ⚠️  Move failed: {e}")
            logger.log_error(file_path, f"Move failed: {e}")
            return 'error', None
    else:
        if verbose:
            print(f"  🔍 Would move to: {dest_path.relative_to(folder_path)}")
        return 'success', dest_path


def cleanup_empty_directories(folder_path: Path, verbose: bool = False, quiet: bool = False):
    """Remove empty directories after organization."""
    if not quiet:
        print("\n🧹 Cleaning up empty directories...")
    
    removed_count = 0
    # Junk files that can be safely removed
    junk_files = {'.DS_Store', 'Thumbs.db', 'desktop.ini', '._.DS_Store'}
    
    for root, dirs, files in os.walk(folder_path, topdown=False):
        root_path = Path(root)
        
        # Skip the root folder itself
        if root_path == folder_path:
            continue
        
        # Remove junk files first
        for file in files:
            if file in junk_files:
                try:
                    (root_path / file).unlink()
                except Exception:
                    pass
        
        # Check if directory is now empty (or only has hidden files)
        try:
            remaining = list(root_path.iterdir())
            # Filter out hidden files for the "empty" check
            non_hidden = [f for f in remaining if not f.name.startswith('.')]
            
            if not non_hidden and not remaining:
                # Truly empty
                root_path.rmdir()
                removed_count += 1
                if verbose:
                    print(f"   Removed: {root_path.relative_to(folder_path)}")
            elif not non_hidden and remaining:
                # Only hidden junk left - remove those too
                for f in remaining:
                    try:
                        f.unlink()
                    except Exception:
                        pass
                # Try removing now
                if not any(root_path.iterdir()):
                    root_path.rmdir()
                    removed_count += 1
                    if verbose:
                        print(f"   Removed: {root_path.relative_to(folder_path)}")
        except Exception:
            pass
    
    if not quiet and removed_count > 0:
        print(f"   Removed {removed_count} empty directories")


def organize_music(folder_path: Path, dry_run: bool = False, verbose: bool = False,
                   skip_organized: bool = True, quiet: bool = False) -> Tuple[int, int]:
    """
    Main function to organize music files in-place.
    Returns tuple of (successful_count, failed_count)
    """
    global _current_logger, _current_log_file, _current_undo_script
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file")
        sys.exit(1)
    
    # Use single timestamp for all files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize cache and logger
    cache_file = folder_path / '.music_organizer_cache.json'
    log_file = folder_path / f'.music_organizer_log_{timestamp}.json'
    undo_script = folder_path / f'undo_organize_{timestamp}.sh'
    
    cache = GeminiCache(cache_file)
    logger = ChangeLogger(log_file)
    
    # Set global references for interrupt handler
    _current_logger = logger
    _current_log_file = log_file
    _current_undo_script = undo_script
    
    print(f"\n🎵 Music Organizer")
    print(f"{'='*50}")
    print(f"Folder: {folder_path}")
    print(f"Mode: {'DRY RUN (no files will be moved)' if dry_run else 'LIVE'}")
    print(f"Skip Organized: {'Yes' if skip_organized else 'No'}")
    print(f"{'='*50}\n")
    
    # Scan for audio files
    print("📂 Scanning for audio files...")
    audio_files = scan_music_folder(folder_path)
    print(f"   Found {len(audio_files)} audio files\n")
    
    if not audio_files:
        print("No audio files found!")
        return 0, 0
    
    successful = 0
    failed = 0
    skipped = 0
    current_album = None
    albums_processed = 0
    
    # First pass: collect file data and identify which need API calls
    print("📋 Analyzing files...")
    files_to_process: List[Dict[str, Any]] = []
    files_needing_api: List[Dict[str, Any]] = []
    
    total_files = len(audio_files)
    for i, file_path in enumerate(audio_files, 1):
        # Skip already organized files
        if skip_organized and is_already_organized(file_path, folder_path):
            if verbose:
                print(f"  ⏭️  {file_path.name} - already organized")
            logger.log_skip(file_path, "Already organized")
            skipped += 1
            continue
        
        # Extract metadata
        metadata = extract_metadata(file_path)
        
        file_data = {
            'file_path': file_path,
            'filename': file_path.name,
            'metadata': metadata,
            'identified': None
        }
        
        # Check if we need Gemini
        has_metadata = all([
            metadata.get('artist'),
            metadata.get('album'),
            metadata.get('title')
        ])
        
        if has_metadata:
            # Use existing metadata
            artist = sanitize_filename(metadata['artist'] or 'Unknown Artist')
            album_artist = sanitize_filename(metadata.get('album_artist') or metadata['artist'] or 'Unknown Artist')
            file_data['identified'] = {
                'artist': artist,
                'album_artist': album_artist,
                'album': sanitize_filename(metadata['album'] or 'Unknown Album'),
                'title': sanitize_filename(metadata['title'] or 'Unknown Title'),
                'is_compilation': False
            }
        else:
            files_needing_api.append(file_data)
        
        files_to_process.append(file_data)
    
    print(f"   {len(files_to_process)} files to organize, {len(files_needing_api)} need AI identification\n")
    
    # Second pass: batch identify files needing API
    if files_needing_api:
        print(f"🤖 Identifying {len(files_needing_api)} files with Gemini AI (batch size: {BATCH_SIZE})...")
        
        for batch_start in range(0, len(files_needing_api), BATCH_SIZE):
            batch = files_needing_api[batch_start:batch_start + BATCH_SIZE]
            batch_end = min(batch_start + BATCH_SIZE, len(files_needing_api))
            
            if not quiet:
                print(f"   Batch {batch_start//BATCH_SIZE + 1}: files {batch_start + 1}-{batch_end}")
            
            # Call batch API
            batch_results = identify_batch_with_gemini(batch, cache, quiet)
            
            # Update file data with results
            for file_data in batch:
                filename = file_data['filename']
                if filename in batch_results:
                    file_data['identified'] = batch_results[filename]
                else:
                    # Fallback
                    metadata = file_data['metadata']
                    artist = metadata.get('artist') or 'Unknown Artist'
                    file_data['identified'] = {
                        'artist': artist,
                        'album_artist': metadata.get('album_artist') or artist,
                        'album': metadata.get('album') or 'Unknown Album',
                        'title': metadata.get('title') or Path(filename).stem,
                        'is_compilation': False
                    }
        
        print()
    
    # Third pass: process all files
    print("📁 Organizing files...")
    for i, file_data in enumerate(files_to_process, 1):
        file_path = file_data['file_path']
        metadata = file_data['metadata']
        identified = file_data['identified']
        
        if identified is None:
            # This shouldn't happen, but fallback just in case
            artist = metadata.get('artist') or 'Unknown Artist'
            identified = {
                'artist': artist,
                'album_artist': metadata.get('album_artist') or artist,
                'album': metadata.get('album') or 'Unknown Album',
                'title': metadata.get('title') or file_path.stem,
                'is_compilation': False
            }
        
        try:
            # Progress output: print when album changes
            album_key = f"{identified.get('album_artist', identified['artist'])}/{identified['album']}"
            if album_key != current_album:
                current_album = album_key
                albums_processed += 1
                if not quiet:
                    print(f"[{i}/{len(files_to_process)}] 💿 {album_key}")
            
            if verbose:
                print(f"  → {file_path.name}")
            
            # Process the file
            status, dest = process_file(file_path, folder_path, metadata, identified,
                                        dry_run, verbose, quiet, logger)
            
            if status == 'success':
                successful += 1
            elif status == 'skip':
                skipped += 1
            else:
                failed += 1
            
        except Exception as e:
            if not quiet:
                print(f"  ❌ Error: {e}")
            logger.log_error(file_path, str(e))
            failed += 1
    
    # Clear progress line in quiet mode
    if quiet:
        print()
    
    # Clean up empty directories
    if not dry_run:
        cleanup_empty_directories(folder_path, verbose, quiet)
    
    # Save logs and generate undo script
    if not dry_run:
        logger.save()
        logger.generate_undo_script(undo_script)
        print(f"\n📝 Change log saved: {log_file.name}")
        print(f"↩️  Undo script saved: {undo_script.name}")
    
    # Clear global references
    _current_logger = None
    _current_log_file = None
    _current_undo_script = None
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Summary")
    print(f"{'='*50}")
    print(f"   💿 Albums: {albums_processed}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Total files: {len(audio_files)}")
    
    if dry_run:
        print(f"\n💡 This was a DRY RUN. Run without --dry-run to actually organize files.")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Organize music files into Artist/Album/Song structure using Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python music_organizer.py --folder ~/Music
  python music_organizer.py --folder ./my_music --dry-run
  python music_organizer.py --folder ./music --verbose
  python music_organizer.py --folder ./music --no-skip  # Re-process all files
        """
    )
    
    parser.add_argument(
        '--folder', '-f',
        type=str,
        required=True,
        help='Music folder to scan and organize in-place'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Preview changes without moving any files'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output for each file'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Process all files, even if already organized'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output, show progress every 50 files'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder).resolve()
    
    # Validate folder
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        sys.exit(1)
    
    # Run organizer
    organize_music(folder_path, args.dry_run, args.verbose, skip_organized=not args.no_skip, quiet=args.quiet)


if __name__ == "__main__":
    main()
