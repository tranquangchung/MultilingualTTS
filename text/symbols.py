""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text.dictionary import cmudict, dutch, french, german, \
    indonesian, italian, japanese, korean, pinyin, \
    polish, portuguese, russian, spanish, vietnamese

import pdb

# _pad = "_"
# _punctuation = "!'(),.:;? "
# _special = "-"
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _pinyin = ["@" + s for s in pinyin.valid_symbols]
# _english = ["@" + s for s in cmudict.valid_symbols]
# _indonesian = ["@" + s for s in indonesian.valid_symbols]
# _japanese = ["@" + s for s in japanese.valid_symbols]
# _korean = ["@" + s for s in korean.valid_symbols]
# _vietnamese = ["@" + s for s in vietnamese.valid_symbols]
# _all_letters = _pinyin + _english + _indonesian + _japanese + _korean + _vietnamese

#################### For all languages ####################
# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_english = ["@" + s for s in cmudict.valid_symbols]
_dutch = ["@" + s for s in dutch.valid_symbols]
_french = ["@" + s for s in french.valid_symbols]
_german = ["@" + s for s in german.valid_symbols]
_indonesian = ["@" + s for s in indonesian.valid_symbols]
_italian = ["@" + s for s in italian.valid_symbols]
_korean = ["@" + s for s in korean.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]
_polish = ["@" + s for s in polish.valid_symbols]
_portuguese = ["@" + s for s in portuguese.valid_symbols]
_russian = ["@" + s for s in russian.valid_symbols]
_spanish = ["@" + s for s in spanish.valid_symbols]
_vietnamese = ["@" + s for s in vietnamese.valid_symbols]
_japanese = ["@" + s for s in japanese.valid_symbols]
# #
_all_letters = _english + _dutch + _french + _german + _indonesian + _italian + \
    _korean + _pinyin + _polish + _portuguese + _russian + _spanish + _vietnamese + _japanese
#################### For only Vietnamese ####################
# _vietnamese = ["@" + s for s in vietnamese.valid_symbols]
# _all_letters = _vietnamese
#############################################################
#################### For only Chinese ####################
# _pinyin = ["@" + s for s in pinyin.valid_symbols]
# _all_letters = _pinyin
####################################################
#################### For only Indonesian ####################
# _indonesian = ["@" + s for s in indonesian.valid_symbols]
# _all_letters = _indonesian
#############################################################
#################### For only Korean ####################
# _korean = ["@" + s for s in korean.valid_symbols]
# _all_letters = _korean
#############################################################
#################### For only Japanese ####################
# _japanese = ["@" + s for s in japanese.valid_symbols]
# _all_letters = _japanese
#############################################################
#################### For only English ####################
# _english = ["@" + s for s in cmudict.valid_symbols]
# _all_letters = _english
#############################################################


symbols = (
    _all_letters
    + _silences
    )
# print(symbols)
# print(len(symbols))
# print("*"*20)