import pandas as pd


def summary_statistics(state):

    data = pd.read_csv('us-states.csv')
    state_data = data[data['state'] == state]
    state_cases = state_data['cases']

    minimum = state_cases.min()
    maximum = state_cases.max()
    mean = state_cases.mean()
    std = state_cases.std()

    output = {
        'Minimum': minimum,
        'Maximum': maximum,
        'Standard Deviation': std,
        'Mean': mean,
    }

    return output


def main():

    state = input('Regarding which state would you like COVID case statistics?: ')
    statistics = summary_statistics(state)

    # The output should contain Minimum, Maximum, Standard deviation and Mean, each data in separate line in below given order:
    # Minimum:
    # Maximum:
    # Mean:
    # Standard Deviation:

    for k, v in statistics.items():
        print("%s: %s" % (k, v))

    return statistics


if __name__ == "__main__":
    main()