from langchain_text_splitters import CharacterTextSplitter

with open('data/data_uet_vnu.txt', 'r', encoding='utf-8') as file:
    document = file.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=100
)
texts = text_splitter.split_text(document)

with open('data/chunked_data.txt', 'w', encoding='utf-8') as output_file:
    for i, text in enumerate(texts, 1):
        output_file.write(f"Chunk {i}:\n{text}\n\n")

print("Results have been saved to data/chunked_data.txt")