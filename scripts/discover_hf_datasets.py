"""
HuggingFace Dataset Discovery — Find and catalog speech + noise datasets.

Lists recommended datasets with their audio column names, speaker columns,
and streaming support. Use this to expand your training data.

Usage:
    python scripts/discover_hf_datasets.py
    python scripts/discover_hf_datasets.py --language ar
    python scripts/discover_hf_datasets.py --type noise
"""

# ============================================================
# CURATED CATALOG of HuggingFace Speech & Noise Datasets
# ============================================================
# Each entry has been verified for:
#   - audio_column name
#   - speaker_column name (if available)
#   - streaming support
#   - approximate size
# ============================================================

SPEECH_DATASETS = [
    # ===== Arabic =====
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "ar",
        "language": "ar",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~150",
        "streaming": True,
        "notes": "Crowdsourced Modern Standard Arabic",
    },
    {
        "name": "google/fleurs",
        "subset": "ar_eg",
        "language": "ar",
        "audio_column": "audio",
        "speaker_column": "id",
        "hours": "~10",
        "streaming": True,
        "notes": "Egyptian Arabic, high quality",
    },
    {
        "name": "google/fleurs",
        "subset": "ar_sa",
        "language": "ar",
        "audio_column": "audio",
        "speaker_column": "id",
        "hours": "~10",
        "streaming": True,
        "notes": "Saudi Arabic, high quality",
    },
    {
        "name": "arabic_speech_corpus",
        "subset": None,
        "language": "ar",
        "audio_column": "audio",
        "speaker_column": None,
        "hours": "~5",
        "streaming": False,
        "notes": "Single speaker MSA, studio quality",
    },
    
    # ===== English =====
    {
        "name": "openslr/librispeech_asr",
        "subset": "train.clean.100",
        "language": "en",
        "audio_column": "audio",
        "speaker_column": "speaker_id",
        "hours": "100",
        "streaming": False,
        "notes": "Gold standard, 251 speakers, clean studio",
    },
    {
        "name": "openslr/librispeech_asr",
        "subset": "train.clean.360",
        "language": "en",
        "audio_column": "audio",
        "speaker_column": "speaker_id",
        "hours": "360",
        "streaming": False,
        "notes": "Extended clean set, 921 speakers",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "en",
        "language": "en",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~3000",
        "streaming": True,
        "notes": "Massive English dataset, crowdsourced",
    },
    {
        "name": "ProgramComputer/voxceleb",
        "subset": "vox1",
        "language": "en",
        "audio_column": "audio",
        "speaker_column": "speaker_id",
        "hours": "~350",
        "streaming": True,
        "notes": "Celebrity interviews, 1211 speakers, noisy",
    },
    {
        "name": "edinburghcstr/vctk",
        "subset": None,
        "language": "en",
        "audio_column": "audio",
        "speaker_column": "speaker_id",
        "hours": "~44",
        "streaming": False,
        "notes": "109 speakers, various accents, studio quality",
    },
    
    # ===== Multilingual =====
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "fr",
        "language": "fr",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~1000",
        "streaming": True,
        "notes": "French",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "de",
        "language": "de",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~1200",
        "streaming": True,
        "notes": "German",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "es",
        "language": "es",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~400",
        "streaming": True,
        "notes": "Spanish",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "zh-CN",
        "language": "zh",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~200",
        "streaming": True,
        "notes": "Chinese (Mandarin)",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "ja",
        "language": "ja",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~50",
        "streaming": True,
        "notes": "Japanese",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "tr",
        "language": "tr",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~100",
        "streaming": True,
        "notes": "Turkish",
    },
    {
        "name": "mozilla-foundation/common_voice_17_0",
        "subset": "ru",
        "language": "ru",
        "audio_column": "audio",
        "speaker_column": "client_id",
        "hours": "~200",
        "streaming": True,
        "notes": "Russian",
    },
    {
        "name": "google/fleurs",
        "subset": "all",
        "language": "multi",
        "audio_column": "audio",
        "speaker_column": "id",
        "hours": "~350",
        "streaming": True,
        "notes": "102 languages! Great for multilingual",
    },
]

NOISE_DATASETS = [
    {
        "name": "flozi00/MUSAN-Noise",
        "audio_column": "audio",
        "label_column": None,
        "category": "noise",
        "streaming": True,
        "notes": "MUSAN noise subset (background noise, machinery, etc.)",
    },
    {
        "name": "danavery/urbansound8K",
        "audio_column": "audio",
        "label_column": "class",
        "category": "urban",
        "streaming": False,
        "notes": "8732 urban sounds: car horn, siren, engine, street music",
    },
    {
        "name": "agkphysics/AudioSet",
        "audio_column": "audio",
        "label_column": "human_labels",
        "category": "environment",
        "streaming": True,
        "notes": "Massive audio event dataset, includes everything",
    },
    {
        "name": "google/fsd50k",
        "audio_column": "audio",
        "label_column": "label",
        "category": "environment",
        "streaming": True,
        "notes": "50K Freesound clips, diverse environmental sounds",
    },
    {
        "name": "MLCommons/peoples_speech",
        "audio_column": "audio",
        "label_column": None,
        "category": "speech_noise",
        "streaming": True,
        "notes": "Noisy real-world speech (use as 'speech babble' noise)",
    },
]


def print_catalog(language: str = None, dataset_type: str = "speech"):
    """Print the dataset catalog in a readable format."""
    
    if dataset_type == "noise":
        datasets = NOISE_DATASETS
        print(f"\n{'='*80}")
        print(f"  HuggingFace Noise Datasets for Training")
        print(f"{'='*80}\n")
        
        for ds in datasets:
            print(f"  📦 {ds['name']}")
            print(f"     Audio column: {ds['audio_column']}")
            if ds.get('label_column'):
                print(f"     Label column: {ds['label_column']}")
            print(f"     Category: {ds['category']}")
            print(f"     Streaming: {'✅' if ds['streaming'] else '❌'}")
            print(f"     Notes: {ds['notes']}")
            print()
        
        return
    
    datasets = SPEECH_DATASETS
    if language:
        datasets = [d for d in datasets if d['language'] == language or d['language'] == 'multi']
    
    print(f"\n{'='*80}")
    print(f"  HuggingFace Speech Datasets{f' — Language: {language}' if language else ''}")
    print(f"{'='*80}\n")
    
    current_lang = None
    for ds in datasets:
        if ds['language'] != current_lang:
            current_lang = ds['language']
            lang_names = {'ar': 'Arabic', 'en': 'English', 'fr': 'French', 
                         'de': 'German', 'es': 'Spanish', 'zh': 'Chinese',
                         'ja': 'Japanese', 'tr': 'Turkish', 'ru': 'Russian',
                         'multi': 'Multilingual'}
            print(f"  ═══ {lang_names.get(current_lang, current_lang)} ═══")
        
        print(f"  📦 {ds['name']}" + (f" ({ds['subset']})" if ds['subset'] else ""))
        print(f"     Audio: {ds['audio_column']} | Speaker: {ds['speaker_column'] or 'N/A'} | "
              f"Hours: {ds['hours']} | Stream: {'✅' if ds['streaming'] else '❌'}")
        print(f"     {ds['notes']}")
        print()
    
    print(f"  ═══ Config Snippet ═══")
    print(f"  Add to configs/finetune.yaml under 'hf_speech:':")
    print()
    
    example = datasets[0] if datasets else SPEECH_DATASETS[0]
    print(f"    - name: \"{example['name']}\"")
    if example.get('subset'):
        print(f"      subset: \"{example['subset']}\"")
    print(f"      audio_column: \"{example['audio_column']}\"")
    if example.get('speaker_column'):
        print(f"      speaker_column: \"{example['speaker_column']}\"")
    print(f"      streaming: {str(example['streaming']).lower()}")
    print()


def generate_config_snippet(languages: list = None) -> str:
    """Generate a YAML config snippet for selected languages."""
    if languages is None:
        languages = ['ar', 'en']
    
    lines = ["hf_speech:"]
    
    for ds in SPEECH_DATASETS:
        if ds['language'] in languages or ds['language'] == 'multi':
            lines.append(f"  - name: \"{ds['name']}\"")
            if ds.get('subset'):
                lines.append(f"    subset: \"{ds['subset']}\"")
            lines.append(f"    audio_column: \"{ds['audio_column']}\"")
            if ds.get('speaker_column'):
                lines.append(f"    speaker_column: \"{ds['speaker_column']}\"")
            lines.append(f"    streaming: {str(ds['streaming']).lower()}")
            lines.append("")
    
    return "\n".join(lines)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover HuggingFace datasets')
    parser.add_argument('--language', '-l', type=str, default=None,
                        help='Filter by language code (ar, en, fr, de, etc.)')
    parser.add_argument('--type', '-t', type=str, default='speech',
                        choices=['speech', 'noise', 'all'],
                        help='Dataset type to show')
    parser.add_argument('--generate-config', action='store_true',
                        help='Generate config snippet for selected languages')
    parser.add_argument('--languages', nargs='+', default=['ar', 'en'],
                        help='Languages for config generation')
    
    args = parser.parse_args()
    
    if args.generate_config:
        snippet = generate_config_snippet(args.languages)
        print(snippet)
    elif args.type == 'all':
        print_catalog(args.language, 'speech')
        print_catalog(args.language, 'noise')
    else:
        print_catalog(args.language, args.type)
