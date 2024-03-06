# This code was created with the help of ChatGPT 4.0
# The prompt was: "i have a csv file with the filename breast-cancer-wisconsin.csv. 
#                 "i want to load the file in R and remove all rows with missing values in any column."
#                 "please write code in R to accomplish this. the missing values are denoted by the character "?""

# Load the CSV file
data <- read.csv("../data/breast-cancer-wisconsin.csv", na.strings="?")

# Remove rows with missing values
clean_data <- na.omit(data)

# View the cleaned data
print(clean_data)
