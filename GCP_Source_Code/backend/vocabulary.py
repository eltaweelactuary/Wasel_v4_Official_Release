"""
Wasel v4 Pro: Dynamic Vocabulary Manager
Replaces hardcoded 9-word vocabulary from v3 with an extensible system.
"""

from typing import Dict, Optional

# ─── Core PSL Vocabulary (Verified in pk-dictionary-mapping.json) ───
PSL_CORE: Dict[str, str] = {
    "good": "اچھا",      # pk-hfad-1_good
    "bad": "برا",         # pk-hfad-1_bad
    "apple": "سیب",      # pk-hfad-1_apple
    "world": "دنیا",     # pk-hfad-1_world
    "pakistan": "پاکستان", # pk-hfad-1_pakistan
    "mother": "ماں",     # pk-hfad-1_mama
    "father": "باپ",     # pk-hfad-1_papa
    "help": "مدد",       # pk-hfad-1_help
    "home": "گھر",       # pk-hfad-1_house
}

# ─── Extended Vocabulary (Awaiting SLT Library Support) ───
PSL_EXTENDED: Dict[str, str] = {
    "hello": "ہیلو",
    "salam": "سلام",
    "water": "پانی",
    "food": "کھانا",
    "thanks": "شکریہ",
    "yes": "ہاں",
    "no": "نہیں",
    "please": "براہ کرم",
    "sorry": "معاف کریں",
    "school": "اسکول",
    "doctor": "ڈاکٹر",
    "friend": "دوست",
    "work": "کام",
    "love": "محبت",
    "is": "ہے",
}


class VocabularyManager:
    """
    Manages the PSL vocabulary with support for:
    - Core vocabulary (verified against SLT library)
    - Extended vocabulary (pending library support)
    - Custom words added at runtime
    """
    
    def __init__(self, include_extended: bool = True):
        self.words: Dict[str, str] = dict(PSL_CORE)
        if include_extended:
            self.words.update(PSL_EXTENDED)
        self._custom: Dict[str, str] = {}
    
    def add_word(self, english: str, urdu: str):
        """Add a custom word at runtime."""
        self._custom[english.lower()] = urdu
        self.words[english.lower()] = urdu
    
    def get_urdu(self, word: str) -> Optional[str]:
        """Lookup Urdu translation for an English word."""
        return self.words.get(word.lower())
    
    def get_all(self) -> Dict[str, str]:
        """Return full vocabulary."""
        return dict(self.words)
    
    def get_core(self) -> Dict[str, str]:
        """Return only core verified vocabulary."""
        return dict(PSL_CORE)
    
    @property
    def size(self) -> int:
        return len(self.words)
    
    def __contains__(self, word: str) -> bool:
        return word.lower() in self.words
    
    def __len__(self) -> int:
        return len(self.words)
