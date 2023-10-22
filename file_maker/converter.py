class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def create_document_from_string(string):
    parts = string.split(", metadata=")
    page_content = parts[0]
    metadata = eval(parts[1]) 
    return Document(page_content, metadata)

with open('input.txt', 'r') as file:
    contents = file.readlines()

documents = [create_document_from_string(content) for content in contents]

with open('output.txt', 'w') as out_file:
    for document in documents:
        out_file.write(document.page_content.split('=')[1].strip().replace('\\', '').replace('\n', '').replace('\r', '').replace('\r\n', ''))

