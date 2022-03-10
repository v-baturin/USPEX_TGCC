from .. import __version__

pic = '''
 _   _ ___ _ __   _____  __  _ __  _   _
| | | / __| '_ \ / _ \ \/ / | '_ \| | | |
| |_| \__ \ |_) |  __/>  < _| |_) | |_| |
 \__,_|___/ .__/ \___/_/\_(_) .__/ \__, |
          |_|               |_|    |___/
'''
pic2 = '''
 _|    _|     _|_|_|   _|_|_|     _|_|_|_|   _|      _|
 _|    _|   _|         _|    _|   _|           _|  _|
 _|    _|     _|_|     _|_|_|     _|_|_|         _|
 _|    _|         _|   _|         _|           _|  _|
   _|_|     _|_|_|     _|         _|_|_|_|   _|      _|
'''


def createHeader(description, debug=False):
    """
    The function is to create headers for USPEX, META, etc. in OUTPUT.txt
    :param description: a description to be placed to OUTPUT.txt.
    :param debug: debug mode.
    :return formatted_rows: the formatted text as a list.
    """

    width = 80
    text = [__version__, '', description, 'more info at http://uspex-team.org', '']
    if debug:
        print('Text: %s' % text)

    # Get the maximum width from the picture file:
    content = pic.split('\n')

    max_len = 0
    rows = []
    for i in range(len(content)):
        current_row = content[i].rstrip()
        rows.append(current_row)
        max_len = max(max_len, len(current_row))

    # ---> Format picture:
    line_len = width
    formatted_rows = []
    delimiter = '|'
    delimiter2 = '-'
    separator = '*' + delimiter2 * (line_len - 2) + '*'

    # Beginning separator:
    formatted_rows.append(separator)

    blanks_before = int(round(line_len / 2.0 - max_len / 2.0 - 1))
    blanks_after = 0
    for i in range(len(rows)):
        blanks_after = line_len - (1 + blanks_before + len(rows[i])) - 1
        new_row = delimiter + ' ' * blanks_before + rows[i] + ' ' * blanks_after + delimiter
        formatted_rows.append(new_row)

    # ---> Format text:
    for i in range(len(text)):
        blanks_before = int(round(line_len / 2.0 - len(text[i]) / 2.0 - 1))
        blanks_after = line_len - (1 + blanks_before + len(text[i])) - 1
        new_row = delimiter + ' ' * blanks_before + text[i] + ' ' * blanks_after + delimiter
        formatted_rows.append(new_row)

    # Ending separator:
    formatted_rows.append(separator)

    return formatted_rows

def createHeader_wrap(text, position='center'):
    '''
    The function wraps the specified text to a table.
    :param text: the text to wrap.
    :param position: the alignment position (centered by default).
    :return formatted_rows : the formatted text represented as a list.
    '''

    # ---> Format text:
    width = 80
    line_len = width
    formatted_rows = []
    delimiter = '|'
    delimiter2 = '-'
    separator = '*' + delimiter2 * (line_len - 2) + '*'

    # Beginning separator:
    formatted_rows.append(separator)

    blanks_before = 2
    for i in range(len(text)):
        if position == 'center':
            blanks_before = int(round(line_len / 2.0 - len(text[i]) / 2.0 - 1))
        blanks_after = line_len - (1 + blanks_before + len(text[i])) - 1
        new_row = delimiter + ' ' * blanks_before + text[i] + ' ' * blanks_after + delimiter
        formatted_rows.append(new_row)

    # Ending separator:
    formatted_rows.append(separator)

    return formatted_rows

