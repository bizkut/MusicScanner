# 🎵 Music Organizer

A Python script that scans a music folder, identifies songs/albums/artists using **Google Gemini AI**, and automatically reorganizes them into a clean `Artist/Album/Song` folder structure.

## ✨ Features

- **AI-Powered Identification**: Uses Google Gemini API to identify artist, album, and song title from filenames when metadata is missing
- **Metadata Extraction**: Reads existing ID3/audio tags using mutagen before falling back to AI
- **Smart Organization**: Creates a clean folder hierarchy: `Artist → Album → Songs`
- **Safe Operations**: Copy-verify-delete approach ensures no data loss
- **Multiple Format Support**: MP3, FLAC, WAV, M4A, AAC, OGG, WMA, OPUS
- **Dry Run Mode**: Preview changes before actually moving files
- **Duplicate Handling**: Automatically handles duplicate filenames
- **Auto Cleanup**: Removes empty folders after reorganization

## 📁 Output Structure

```
Music Folder/
├── Taylor Swift/
│   ├── 1989/
│   │   ├── 01 - Shake It Off.mp3
│   │   └── 02 - Blank Space.mp3
│   └── Midnights/
│       └── 01 - Anti-Hero.mp3
├── The Beatles/
│   └── Abbey Road/
│       └── 01 - Come Together.mp3
└── Unknown Artist/
    └── Singles/
        └── random_song.mp3
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MusicScanner.git
cd MusicScanner
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Run the Organizer

```bash
# Preview changes (recommended first)
python music_organizer.py --folder ~/Music --dry-run --verbose

# Actually organize files
python music_organizer.py --folder ~/Music
```

## 📖 Usage

```
usage: music_organizer.py [-h] --folder FOLDER [--dry-run] [--verbose]

Organize music files into Artist/Album/Song structure using Gemini AI

options:
  -h, --help            show this help message and exit
  --folder, -f FOLDER   Music folder to scan and organize in-place
  --dry-run, -d         Preview changes without moving any files
  --verbose, -v         Show detailed output for each file

Examples:
  python music_organizer.py --folder ~/Music
  python music_organizer.py --folder ./my_music --dry-run
  python music_organizer.py --folder ./music --verbose
```

## 🔒 Safety Features

1. **Copy-Verify-Delete**: Files are copied first, verified (size check), then original is deleted only after successful copy
2. **Dry Run Mode**: Always preview with `--dry-run` before actual reorganization
3. **Verification Failed**: If copy verification fails, original file is kept untouched
4. **Duplicate Protection**: Automatically renames duplicates instead of overwriting

## 🔧 Requirements

- Python 3.8+
- Google Gemini API key

### Dependencies

- `google-generativeai` - Gemini API SDK
- `mutagen` - Audio metadata extraction
- `python-dotenv` - Environment variable management

## 📄 License

MIT License - feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
