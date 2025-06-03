from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter, CharacterTextSplitter
from langchain_text_splitters import SpacyTextSplitter

with open('input.txt', 'r', encoding='utf-8') as file:
    document = file.read()

# Cách này đang bị chia theo token tức là chunk được nửa dòng thì ngắt
# text_splitter = TokenTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=512, chunk_overlap=51
# )
# texts = text_splitter.split_text(document)

# Cách này có vẻ ổn do nó tách để đảm bảo lấy hết cả bảng
# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=512, chunk_overlap=51
# )
# texts = text_splitter.split_text(document)

# Cách này đang được dùng tạm nó catch được nhiều row trong bảng tuy nhiên chỉ phù hợp khi chunk size lớn
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1024,
    chunk_overlap=102,
)
texts = text_splitter.split_text(document)


with open('output.txt', 'w', encoding='utf-8') as output_file:
    for i, text in enumerate(texts, 1):
        output_file.write(f"Chunk {i}:\n{text}\n\n")

print(len(texts))
print("Results have been saved.")