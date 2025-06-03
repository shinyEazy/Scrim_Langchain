from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter

with open('input.md', 'r', encoding='utf-8') as file:
    document = file.read()

# text_splitter = TokenTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=512, chunk_overlap=51
# )
# texts = text_splitter.split_text(document)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=512, chunk_overlap=51
)
texts = text_splitter.split_text(document)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=51)
# texts = text_splitter.split_text(document)

with open('output.txt', 'w', encoding='utf-8') as output_file:
    for i, text in enumerate(texts, 1):
        output_file.write(f"Chunk {i}:\n{text}\n\n")

print(len(texts))
print("Results have been saved.")