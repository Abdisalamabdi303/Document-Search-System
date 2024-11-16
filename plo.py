import pandas as pd
import time
import random
import matplotlib.pyplot as plt

class DocumentSearchSystem:
    def __init__(self, dataframe):
        # Store the dataframe and build search indices
        self.data = dataframe
        self.id_index = self.build_id_index(dataframe)
        self.inverted_index = self.build_inverted_index(dataframe)

    def build_id_index(self, dataframe):
        """Builds an index for exact document ID search."""
        return {row['Document ID']: row for _, row in dataframe.iterrows()}
    
    def build_inverted_index(self, dataframe):
        """Builds an inverted index for partial metadata search on title and author."""
        inverted_index = {}
        for _, row in dataframe.iterrows():
            # Tokenize title and author
            tokens = str(row['Document Title']).split() + str(row['Author']).split()
            document_id = row['Document ID']
            for token in tokens:
                token = token.lower()  # Normalize to lowercase
                if token in inverted_index:
                    inverted_index[token].add(document_id)
                else:
                    inverted_index[token] = {document_id}
        return inverted_index

    def search_by_id(self, document_id):
        """Searches for a document by its exact Document ID."""
        return self.id_index.get(document_id, None)

    def search_by_metadata(self, keyword):
        """Searches for documents by partial metadata (title/author keyword)."""
        keyword = keyword.lower()
        matching_ids = self.inverted_index.get(keyword, set())
        # Return documents matching the IDs found in the inverted index
        return [self.id_index[doc_id] for doc_id in matching_ids]

    def measure_performance(self, document_id, keyword, num_trials=10):
        """Measure the average search time for exact ID and partial metadata search."""
        # Measure time for exact ID search
        start_time = time.time()
        for _ in range(num_trials):
            self.search_by_id(document_id)
        id_search_time = (time.time() - start_time) / num_trials

        # Measure time for partial metadata search
        start_time = time.time()
        for _ in range(num_trials):
            self.search_by_metadata(keyword)
        metadata_search_time = (time.time() - start_time) / num_trials

        return id_search_time, metadata_search_time

# Load your provided dataset
file_path = r'C:\Users\hp\Downloads\Datasets_SENG 313\document_archive3.csv'  # path of file

try:
    document_df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
    exit()

# Initialize search system with the real dataset
search_system = DocumentSearchSystem(document_df)

# Choose random document ID and keyword from dataset for testing
sample_row = document_df.sample()
sample_document_id = sample_row['Document ID'].values[0]
sample_keyword = random.choice(sample_row['Document Title'].values[0].split())  # Random keyword from title

# Measure performance for the real dataset
id_time, metadata_time = search_system.measure_performance(sample_document_id, sample_keyword)

# Display results
print(f"Dataset Size: {len(document_df)} records")
print(f"Exact ID Search Time (s): {id_time:.6f}")
print(f"Partial Metadata Search Time (s): {metadata_time:.6f}")

# Performance Testing Across Different Dataset Sizes
sizes = [100, 1000, 10000]  # Example dataset sizes
id_times = []
metadata_times = []

for size in sizes:
    file_path = f'C:\\Users\\hp\\Downloads\\Datasets_SENG 313\\document_archive3.csv'  # Make sure these files exist
    try:
        document_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        continue
    
    search_system = DocumentSearchSystem(document_df)
    
    # Randomly select ID and keyword for testing
    sample_row = document_df.sample()
    sample_document_id = sample_row['Document ID'].values[0]
    sample_keyword = random.choice(sample_row['Document Title'].values[0].split())
    
    id_time, metadata_time = search_system.measure_performance(sample_document_id, sample_keyword)
    
    id_times.append(id_time)
    metadata_times.append(metadata_time)

# Plotting the performance comparison
plt.plot(sizes, id_times, label='Exact ID Search Time', marker='o')
plt.plot(sizes, metadata_times, label='Partial Metadata Search Time', marker='o')
plt.xlabel('Dataset Size (records)')
plt.ylabel('Average Search Time (s)')
plt.title('Search Performance Comparison')
plt.legend()
plt.grid()
plt.show()
