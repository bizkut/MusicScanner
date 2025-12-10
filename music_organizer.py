#!/usr/bin/env python3
"""
Music Organizer Script
Scans a music folder, identifies songs/albums/artists using Gemini API,
and reorganizes them into Artist/Album/Song folder structure.
"""

import os
import sys
import argparse
import shutil
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from dotenv import load_dotenv
import google.generativeai as genai
from mutagen import File as MutagenFile
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError

# Load environment variables from .env file
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPPORTED_FORMATS = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma', '.opus'}

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def sanitize_filename(name: str) -> str:
    """Remove invalid characters from filename/folder name."""
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
    metadata = {
        'artist': None,
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


def identify_with_gemini(filename: str, existing_metadata: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Use Gemini API to identify song, artist, and album from filename and partial metadata."""
    
    # Build context for Gemini
    context_parts = [f"Filename: {filename}"]
    if existing_metadata.get('artist'):
        context_parts.append(f"Known Artist: {existing_metadata['artist']}")
    if existing_metadata.get('album'):
        context_parts.append(f"Known Album: {existing_metadata['album']}")
    if existing_metadata.get('title'):
        context_parts.append(f"Known Title: {existing_metadata['title']}")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Analyze this music file and identify the artist, album, and song title.
If information is already provided, validate and use it. If not, try to identify from the filename.

{context}

Respond ONLY with a valid JSON object in this exact format (no markdown, no explanation):
{{"artist": "Artist Name", "album": "Album Name", "title": "Song Title"}}

If you cannot determine a field with confidence, use "Unknown" for that field.
For the album, if you're not sure, you can use "Singles" or the artist's name.
"""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response - remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        result = json.loads(response_text)
        return {
            'artist': sanitize_filename(result.get('artist', 'Unknown Artist')),
            'album': sanitize_filename(result.get('album', 'Unknown Album')),
            'title': sanitize_filename(result.get('title', 'Unknown Title'))
        }
    except json.JSONDecodeError as e:
        print(f"  Warning: Could not parse Gemini response for {filename}: {e}")
        print(f"  Response was: {response_text[:200] if 'response_text' in dir() else 'N/A'}")
    except Exception as e:
        print(f"  Warning: Gemini API error for {filename}: {e}")
    
    # Fallback to existing metadata or filename
    return {
        'artist': existing_metadata.get('artist') or 'Unknown Artist',
        'album': existing_metadata.get('album') or 'Unknown Album',
        'title': existing_metadata.get('title') or Path(filename).stem
    }


def scan_music_folder(source_path: Path) -> List[Path]:
    """Recursively scan folder for audio files."""
    audio_files = []
    
    for root, dirs, files in os.walk(source_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_FORMATS:
                audio_files.append(file_path)
    
    return audio_files


def organize_music(source_path: Path, output_path: Path, dry_run: bool = False, verbose: bool = False) -> Tuple[int, int]:
    """
    Main function to organize music files.
    Returns tuple of (successful_count, failed_count)
    """
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file")
        sys.exit(1)
    
    print(f"\n🎵 Music Organizer")
    print(f"{'='*50}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Mode: {'DRY RUN (no files will be moved)' if dry_run else 'LIVE'}")
    print(f"{'='*50}\n")
    
    # Scan for audio files
    print("📂 Scanning for audio files...")
    audio_files = scan_music_folder(source_path)
    print(f"   Found {len(audio_files)} audio files\n")
    
    if not audio_files:
        print("No audio files found!")
        return 0, 0
    
    successful = 0
    failed = 0
    
    # Process each file
    for i, file_path in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] Processing: {file_path.name}")
        
        try:
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
                identified = identify_with_gemini(file_path.name, metadata)
            else:
                identified = {
                    'artist': sanitize_filename(metadata['artist']),
                    'album': sanitize_filename(metadata['album']),
                    'title': sanitize_filename(metadata['title'])
                }
            
            # Build destination path
            artist_folder = output_path / identified['artist']
            album_folder = artist_folder / identified['album']
            
            # Build new filename
            if metadata.get('track_number'):
                new_filename = f"{metadata['track_number']} - {identified['title']}{file_path.suffix.lower()}"
            else:
                new_filename = f"{identified['title']}{file_path.suffix.lower()}"
            
            dest_path = album_folder / new_filename
            
            if verbose:
                print(f"  → Artist: {identified['artist']}")
                print(f"  → Album: {identified['album']}")
                print(f"  → Title: {identified['title']}")
                print(f"  → Destination: {dest_path}")
            
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
                # Copy file (safer than move, user can delete source later)
                shutil.copy2(file_path, dest_path)
                print(f"  ✅ Copied to: {dest_path.relative_to(output_path)}")
            else:
                print(f"  🔍 Would copy to: {dest_path.relative_to(output_path)}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Summary")
    print(f"{'='*50}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📁 Total: {len(audio_files)}")
    
    if dry_run:
        print(f"\n💡 This was a DRY RUN. Run without --dry-run to actually organize files.")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description="Organize music files into Artist/Album/Song structure using Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python music_organizer.py --source ~/Downloads/Music --output ~/Music/Organized
  python music_organizer.py --source ./messy_music --output ./organized --dry-run
  python music_organizer.py --source ./music --output ./sorted --verbose
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source folder containing unorganized music files'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output folder for organized music structure'
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
    
    args = parser.parse_args()
    
    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    
    # Validate source
    if not source_path.exists():
        print(f"Error: Source folder does not exist: {source_path}")
        sys.exit(1)
    
    if not source_path.is_dir():
        print(f"Error: Source is not a directory: {source_path}")
        sys.exit(1)
    
    # Create output if needed
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Run organizer
    organize_music(source_path, output_path, args.dry_run, args.verbose)


if __name__ == "__main__":
    main()
