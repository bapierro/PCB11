import pandas as pd

# Step 1: Load the Anderson behaviour file
# Adjust the path if needed
behaviour_file = "data/behaviour/CategoryValidationExpt.txt"

df = pd.read_csv(behaviour_file)

# Step 2: Extract only relevant columns
df_subset = df[['SYNSscene', 'SYNSView']]

# Step 3: Drop duplicates to get unique images
unique_images = df_subset.drop_duplicates().sort_values(by=['SYNSscene', 'SYNSView'])

# Step 4: Reset index (optional)
unique_images = unique_images.reset_index(drop=True)

# Step 5: Save to CSV
output_file = "data/behaviour/syns_images_to_download.csv"
unique_images.to_csv(output_file, index=False)

print(f"✅ Done! {len(unique_images)} unique images to download saved to {output_file}")