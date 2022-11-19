import re
import pdb

multipleThousand = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ", "tỷ tỷ"]
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]

_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_number_re = re.compile(r'-?[0-9]+')

def _remove_commas(m):
  return m.group(1).replace(',', ' phẩy ')

def _expand_decimal_point(m):
  return m.group(1).replace('.', ' chấm ')

def readTriple(triple: str, showZeroHundred: bool):
    a, b, c = int(triple[0]), int(triple[1]), int(triple[2])
    if a == 0 and b == 0 and c == 0: return ""
    elif a == 0 and showZeroHundred: return "không trăm " + readPair(b, c)
    elif a == 0 and b == 0: return digits[c]
    elif a == 0 and b != 0: return readPair(b, c)
    else: return digits[a] + " trăm " + readPair(b, c)

def readPair(b: int, c: int):
    if b == 0:
        if c == 0: return ""
        else: return "lẻ " + digits[c]
    if b == 1:
        if c == 0: return "mười"
        elif c == 5: return "mười lăm"
        else: return "mười " + digits[c]
    else:
        if c == 0: return digits[b] + " mươi"
        elif c == 1: return digits[b] + " mươi một"
        elif c == 4: return  digits[b] + " mươi tư"
        elif c == 5: return digits[b] + " mươi lăm"
        else: return digits[b] + " mươi " + digits[c]

def convert(number: str):
    if int(number) == 0: return "không"
    elif int(number) < 0: return "âm " + convert(number[1:])
    else:
        length = len(number)
        group = int(length / 3)
        # padding zero at ahead of string
        number = number.zfill((group+1)*3)
        number_str = ""
        showZeroHundred = group > 0
        for g, unit in enumerate(range(group, -1, -1)):
            triple_number = number[g*3:g*3+3]
            read_triple = readTriple(triple_number, (showZeroHundred and g > 0))
            if read_triple:
                number_str += read_triple + " " + multipleThousand[unit] + " "
    return number_str

def _expand_number(m):
    return convert(m.group(0))

def normalize_numbers_vn(text):
  text = re.sub(_comma_number_re, _remove_commas, text)
  text = re.sub(_decimal_number_re, _expand_decimal_point, text)
  text = re.sub(_number_re, _expand_number, text)
  return text

if __name__ == "__main__":
    tmp = normalize_numbers_vn("987,789,431")
    print(tmp)
