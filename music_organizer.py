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
SUPPORTED_FORMATS = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'}
API_RATE_LIMIT_DELAY = 0.5  # seconds between API calls
SKIP_DIRECTORIES = {'.git', '.svn', '__pycache__', 'node_modules'}
PROGRESS_INTERVAL = 50  # Print progress every N files in quiet mode

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


def identify_with_gemini(filename: str, existing_metadata: Dict[str, Optional[str]], 
                         cache: Optional[GeminiCache] = None) -> Dict[str, Any]:
    """Use Gemini API to identify song, artist, album, and detect compilations."""
    
    # Create cache key from filename and existing metadata
    cache_key = f"{filename}|{existing_metadata.get('artist')}|{existing_metadata.get('album')}"
    
    # Check cache first
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return cached
    
    # Build context for Gemini
    context_parts = [f"Filename: {filename}"]
    if existing_metadata.get('artist'):
        context_parts.append(f"Known Track Artist: {existing_metadata['artist']}")
    if existing_metadata.get('album_artist'):
        context_parts.append(f"Known Album Artist: {existing_metadata['album_artist']}")
    if existing_metadata.get('album'):
        context_parts.append(f"Known Album: {existing_metadata['album']}")
    if existing_metadata.get('title'):
        context_parts.append(f"Known Title: {existing_metadata['title']}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Analyze this music file and identify the track artist, album artist, album, and song title.

{context}

IMPORTANT: Determine if this is a COMPILATION album (soundtracks, "Various Artists", "Now That's What I Call Music", etc.)
- For regular albums: album_artist should be the same as artist
- For compilations: album_artist should be "Various Artists"

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation):
{{"artist": "Track Artist Name", "album_artist": "Album Artist or Various Artists", "album": "Album Name", "title": "Song Title", "is_compilation": true/false}}

If you cannot determine a field with confidence, use "Unknown" for that field.
For the album, if you're not sure, you can use "Singles".
"""

    response_text = ""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Rate limiting AFTER successful call
        time.sleep(API_RATE_LIMIT_DELAY)
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        result = json.loads(response_text)
        identified: Dict[str, Any] = {
            'artist': sanitize_filename(result.get('artist', 'Unknown Artist')),
            'album_artist': sanitize_filename(result.get('album_artist', result.get('artist', 'Unknown Artist'))),
            'album': sanitize_filename(result.get('album', 'Unknown Album')),
            'title': sanitize_filename(result.get('title', 'Unknown Title')),
            'is_compilation': result.get('is_compilation', False)
        }
        
        # Cache the result
        if cache:
            cache.set(cache_key, identified)
        
        return identified
        
    except json.JSONDecodeError as e:
        response_preview = response_text[:200] if response_text else 'N/A'
        print(f"  Warning: Could not parse Gemini response for {filename}: {e}")
        print(f"  Response was: {response_preview}")
    except Exception as e:
        print(f"  Warning: Gemini API error for {filename}: {e}")
    
    # Fallback to existing metadata or filename
    artist = existing_metadata.get('artist') or 'Unknown Artist'
    fallback: Dict[str, Any] = {
        'artist': artist,
        'album_artist': existing_metadata.get('album_artist') or artist,
        'album': existing_metadata.get('album') or 'Unknown Album',
        'title': existing_metadata.get('title') or Path(filename).stem,
        'is_compilation': False
    }
    return fallback


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
        
        # SAFE MOVE: Copy first, verify, then delete original
        # Step 1: Copy file to destination
        shutil.copy2(str(file_path), str(dest_path))
        
        # Step 2: Verify copy succeeded (check file exists and size matches)
        if dest_path.exists() and dest_path.stat().st_size == file_path.stat().st_size:
            # Step 3: Only delete original after verified copy
            file_path.unlink()
            if verbose:
                print(f"  ✅ Moved to: {dest_path.relative_to(folder_path)}")
            logger.log_move(file_path, dest_path, identified)
            return 'success', dest_path
        else:
            # Copy failed - remove incomplete copy, keep original
            if dest_path.exists():
                dest_path.unlink()
            if not quiet:
                print(f"  ⚠️  Copy verification failed, original kept safe")
            logger.log_error(file_path, "Copy verification failed")
            return 'error', None
    else:
        if verbose:
            print(f"  🔍 Would move to: {dest_path.relative_to(folder_path)}")
        return 'success', dest_path


def cleanup_empty_directories(folder_path: Path, verbose: bool = False):
    """Remove empty directories after organization."""
    print("\n🧹 Cleaning up empty directories...")
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for dir_name in dirs:
            dir_path = Path(root) / dir_name
            try:
                if dir_path.exists() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    if verbose:
                        print(f"   Removed empty: {dir_path.relative_to(folder_path)}")
            except Exception:
                pass


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
    current_album = None  # Track current album for progress output
    albums_processed = 0
    
    # Process each file
    total_files = len(audio_files)
    for i, file_path in enumerate(audio_files, 1):
        try:
            # Resume support: skip already organized files
            if skip_organized and is_already_organized(file_path, folder_path):
                if verbose:
                    print(f"  ⏭️  Already in organized structure, skipping")
                logger.log_skip(file_path, "Already organized")
                skipped += 1
                continue
            
            # Extract existing metadata
            metadata = extract_metadata(file_path)
            
            # Check if we need Gemini (missing critical metadata)
            needs_identification = not all([
                metadata.get('artist'),
                metadata.get('album'),
                metadata.get('title')
            ])
            
            if needs_identification:
                if verbose:
                    print(f"  📡 Querying Gemini API...")
                identified = identify_with_gemini(file_path.name, metadata, cache)
            else:
                # Use existing metadata
                artist = sanitize_filename(metadata['artist'] or 'Unknown Artist')
                album_artist = sanitize_filename(metadata.get('album_artist') or metadata['artist'] or 'Unknown Artist')
                identified: Dict[str, Any] = {
                    'artist': artist,
                    'album_artist': album_artist,
                    'album': sanitize_filename(metadata['album'] or 'Unknown Album'),
                    'title': sanitize_filename(metadata['title'] or 'Unknown Title'),
                    'is_compilation': False
                }
            
            # Progress output: print when album changes
            album_key = f"{identified.get('album_artist', identified['artist'])}/{identified['album']}"
            if album_key != current_album:
                current_album = album_key
                albums_processed += 1
                if not quiet:
                    print(f"[{i}/{total_files}] 💿 {album_key}")
                elif verbose:
                    print(f"[{i}/{total_files}] Processing album: {album_key}")
            
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
        cleanup_empty_directories(folder_path, verbose)
    
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
