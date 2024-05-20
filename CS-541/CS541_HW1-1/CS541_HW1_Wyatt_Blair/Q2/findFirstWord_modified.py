# do not change any function names
def compare_words(word1, word2):
    # return 1 if word1 is first alphabetically and 2 if word2 is the first alphabetically

    # if an empty string is passed in, then it's the first alphabetically
    if not word1:
        return 1
    elif not word2:
        return 2

    # only need to look at the first letter
    letter1 = word1[0]
    letter2 = word2[0]

    # use python string comparison logic to determine if first letter of word is less than, greater than, or equal
    if letter1 < letter2:
        return 1

    elif letter2 < letter1:
        return 2

    # in the case that the letters are equal, clip the first letter and recursively call compare_words on the
    # clipped words
    else:
        clipped_word1 = word1[1:]
        clipped_word2 = word2[1:]

        return compare_words(word1=clipped_word1, word2=clipped_word2)


def findAlphabeticallyFirstWord(string, verbose=False):

    if ' ' in string:
        words = string.split(' ')
    else:
        words = string.split(',')

    first_word = words[0]
    for word in words[1:]:

        comparison_dict = {1: first_word, 2: word}

        out = compare_words(first_word, word)
        first_word = comparison_dict[out]

        if verbose:
            print('COMPARING: %s | %s' % tuple(comparison_dict.values()))
            print('WINNER: %s' % first_word)
            print('='*30)

    return first_word


# Ask the user for the input string, separated by a space or comma
string = input('Please enter a list of words, separated by a space or comma, for the function to sort: ')

# Find the first word in alphabetical order
alphabetically_first_word = findAlphabeticallyFirstWord(string)

# Print the result
print(alphabetically_first_word)
