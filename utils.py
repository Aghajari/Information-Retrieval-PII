from datetime import datetime


def binary_search_tuple(a, x):
    """
    Perform a binary search on a list of tuples to find the second value
    associated with a given first value.

    Args:
        a (list of tuples): A sorted list of tuples where each tuple contains
                        a key as the first element and a value as the second element.
        x (any): The key to search for in the list of tuples.

    Returns:
        any: The value associated with the key if the key is found.

    Raises:
        KeyError: If the key is not found in the list.

    Example:
        >>> l = [(1, 'a'), (2, 'b'), (3, 'c')]
        >>> binary_search_tuple(l, 2)
        'b'

        >>> binary_search_tuple(l, 4)
        KeyError: '4 Not found!'
    """
    low = 0
    high = len(a) - 1
    while low < high:
        mid = (low + high) // 2
        if a[mid][0] < x:
            low = mid + 1
        else:
            high = mid

    if a[low][0] == x:
        return a[low][1]
    else:
        raise KeyError(f"{x} Not found!")


def parse_date(date: str, default: datetime):
    try:
        return datetime.strptime(date, '%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        return default
