import json
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os

# --- Configuration ---
input_file_path = r'..\shared task\data\humility_added.json'
output_file_path = r'..\shared task\data\humility_comments_reduced.json' # New file for reduced data
comment_count_threshold = None # We'll determine this after plotting

# --- 1. Load Data ---
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Successfully loaded data from {input_file_path}")
except FileNotFoundError:
    print(f"Error: Input file not found at {input_file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_file_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading data: {e}")
    exit()

# --- 2. Count Comments per User ---
comment_counts = []
user_ids = []
for user_data in data:
    comments = user_data.get('comments')
    user_id = user_data.get('id', 'Unknown') # Get user ID, default to 'Unknown' if missing
    if comments and isinstance(comments, list):
        count = len(comments)
        comment_counts.append(count)
        user_ids.append(user_id)
    else:
        # Handle users with no comments or invalid comment format
        comment_counts.append(0)
        user_ids.append(user_id)
        # print(f"Warning: User {user_id} has no comments or invalid format.") # Optional: log warnings

if not comment_counts:
    print("No user data with comments found. Exiting.")
    exit()

# Convert to pandas Series for easy analysis
comment_counts_series = pd.Series(comment_counts)

# --- 3. Visualize the Distribution ---
print("\nComment count distribution statistics:")
print(comment_counts_series.describe())

plt.figure(figsize=(12, 6))

# Histogram
plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
plt.hist(comment_counts, bins=range(min(comment_counts), max(comment_counts) + 2), edgecolor='black')
plt.title('Distribution of Comment Counts per User (Histogram)')
plt.xlabel('Number of Comments')
plt.ylabel('Number of Users')
plt.grid(axis='y', alpha=0.75)

# Box Plot
plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
plt.boxplot(comment_counts, vert=False, patch_artist=True)
plt.title('Distribution of Comment Counts per User (Box Plot)')
plt.xlabel('Number of Comments')
plt.yticks([1], ['Comment Count']) # Label the y-axis

plt.tight_layout() # Adjust layout to prevent overlap
plt.show()

# --- 4. Identify Outliers and Determine Threshold ---
# After viewing the plots, decide on a reasonable threshold.
# A common approach is mean + 2*std dev, or looking at the box plot's whiskers.
# Let's calculate a suggested threshold based on mean + 2*std dev, but you can adjust this manually
mean_count = comment_counts_series.mean()
std_count = comment_counts_series.std()
suggested_threshold = int(np.ceil(mean_count + 2 * std_count)) # Use ceil to ensure it's an integer

print(f"\nSuggested comment count threshold (Mean + 2*StdDev): {suggested_threshold}")

# --- Manual Threshold Adjustment (Optional) ---
# You can uncomment the line below and set a specific number after viewing the plot
# comment_count_threshold = 10 # Example: Cap at 10 comments per user

# Use the suggested threshold if not manually set
if comment_count_threshold is None:
    comment_count_threshold = suggested_threshold

print(f"Using comment count threshold: {comment_count_threshold}")

# --- 5. Reduce Comments for Outliers ---
modified_data = []
users_reduced_count = 0
total_original_comments = sum(comment_counts)
total_reduced_comments = 0

for user_data in data:
    comments = user_data.get('comments')
    if comments and isinstance(comments, list):
        original_count = len(comments)
        if original_count > comment_count_threshold:
            # Randomly sample comments up to the threshold
            reduced_comments = random.sample(comments, comment_count_threshold)
            user_data['comments'] = reduced_comments
            users_reduced_count += 1
            total_reduced_comments += comment_count_threshold
            # print(f"Reduced comments for user {user_data.get('id', 'Unknown')} from {original_count} to {comment_count_threshold}")
        else:
            # Keep comments as they are
            total_reduced_comments += original_count
    else:
         # Keep users with no comments or invalid format as they are (count 0)
         pass # Their count is already 0, no reduction needed

    modified_data.append(user_data) # Add the potentially modified user data

print(f"\nFinished reducing comments.")
print(f"Number of users whose comments were reduced: {users_reduced_count}")
print(f"Total original comments across all users: {total_original_comments}")
print(f"Total comments after reduction: {total_reduced_comments}")


# --- 6. Save Modified Data ---
try:
    # Ensure the directory exists
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2) # Use indent for readability
    print(f"Successfully saved modified data to {output_file_path}")
except Exception as e:
    print(f"Error saving modified data to {output_file_path}: {e}")

