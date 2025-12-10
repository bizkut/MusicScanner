# 🎵 Music Organizer

A Python script that scans a music folder, identifies songs/albums/artists using **Google Gemini AI**, and automatically reorganizes them into a clean `Artist/Album/Song` folder structure.

## ✨ Features

- **AI-Powered Identification**: Uses Google Gemini API to identify artist, album, and song title from filenames when metadata is missing
- **Metadata Extraction**: Reads existing ID3/audio tags using mutagen before falling back to AI
- **Smart Organization**: Creates a clean folder hierarchy: `Artist → Album → Songs`
- **Multiple Format Support**: MP3, FLAC, WAV, M4A, AAC, OGG, WMA, OPUS
- **Dry Run Mode**: Preview changes before actually moving files
- **Duplicate Handling**: Automatically handles duplicate filenames
- **Safe Operations**: Copies files instead of moving (preserves originals)

## 📁 Output Structure

```
Organized Music/
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
python music_organizer.py --source ~/Downloads/Music --output ~/Music/Organized --dry-run

# Actually organize files
python music_organizer.py --source ~/Downloads/Music --output ~/Music/Organized

# Verbose mode for detailed output
python music_organizer.py --source ~/Downloads/Music --output ~/Music/Organized --verbose
```

## 📖 Usage

```
usage: music_organizer.py [-h] --source SOURCE --output OUTPUT [--dry-run] [--verbose]

Organize music files into Artist/Album/Song structure using Gemini AI

options:
  -h, --help            show this help message and exit
  --source, -s SOURCE   Source folder containing unorganized music files
  --output, -o OUTPUT   Output folder for organized music structure
  --dry-run, -d         Preview changes without moving any files
  --verbose, -v         Show detailed output for each file

Examples:
  python music_organizer.py --source ~/Downloads/Music --output ~/Music/Organized
  python music_organizer.py --source ./messy_music --output ./organized --dry-run
  python music_organizer.py --source ./music --output ./sorted --verbose
```

## 🔧 Requirements

- Python 3.8+
- Google Gemini API key

### Dependencies

- `google-generativeai` - Gemini API SDK
- `mutagen` - Audio metadata extraction
- `python-dotenv` - Environment variable management

## ⚠️ Important Notes

1. **Backup your music** before running without `--dry-run`
2. The script **copies** files instead of moving them to preserve originals
3. Files with complete metadata (artist, album, title) won't trigger API calls
4. API rate limits may apply depending on your Gemini API quota

## 📄 License

MIT License - feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
