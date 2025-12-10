# 🎵 Music Organizer

A Python script that scans a music folder, identifies songs/albums/artists using **Google Gemini AI**, and automatically reorganizes them into a clean `Artist/Album/Song` folder structure.

## ✨ Features

- **AI-Powered Identification** - Uses Gemini API to identify artist, album, and title
- **Batch Processing** - Sends 10 files per API call to reduce rate limiting
- **Retry with Backoff** - Automatic retry on rate limits (429 errors)
- **Compilation Support** - Detects "Various Artists" albums and soundtracks
- **Fast Operations** - Uses instant rename/move (no file copying)
- **Resume Support** - Skips already-organized files
- **Undo Script** - Auto-generates shell script to reverse all changes
- **Change Logging** - JSON log of all operations for auditing
- **Response Caching** - Reduces API calls for repeated runs

## 📁 Output Structure

```
Music/
├── Taylor Swift/
│   └── 1989/
│       ├── 01 - Shake It Off.mp3
│       └── 02 - Blank Space.mp3
└── Various Artists/
    └── Guardians of the Galaxy/
        ├── 01 - Redbone - Come and Get Your Love.mp3
        └── 02 - Blue Swede - Hooked on a Feeling.mp3
```

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/bizkut/MusicScanner.git
cd MusicScanner
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run (dry-run first!)
python music_organizer.py --folder ~/Music --dry-run
python music_organizer.py --folder ~/Music
```

## 📖 Usage

```
python music_organizer.py --folder PATH [OPTIONS]

Options:
  --folder, -f    Music folder to organize (required)
  --dry-run, -d   Preview changes without moving files
  --verbose, -v   Show detailed per-file output
  --quiet, -q     Minimal output
  --no-skip       Re-process already organized files
```

## ⚙️ Configuration

Create a `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash  # Optional, this is the default
```

## 📄 Generated Files

After running:
```
Music/
├── .music_organizer_cache.json    # API response cache
├── .music_organizer_log_*.json    # Change log
└── undo_organize_*.sh             # Undo script
```

## 🔧 Requirements

- Python 3.8+
- [Gemini API key](https://aistudio.google.com/app/apikey)

## 📄 License

MIT
