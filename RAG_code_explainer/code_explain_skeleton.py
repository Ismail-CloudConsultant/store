from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
import ast
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda



import os
import dotenv

dotenv.load_dotenv()
# Load Python files from the 'codes' directory
loader = DirectoryLoader(
    path="codes",
    glob="*.py",
    loader_cls=TextLoader
)

docs = loader.load()
print("Loaded:", len(docs), "Python files")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

final_chunks = []
for fn_doc in docs:
    small = recursive_splitter.split_documents([fn_doc])
    final_chunks.extend(small)

print("Final chunk count:", len(final_chunks))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(final_chunks, embeddings)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 30, "fetch_k": 60}
)
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

class FlowFormatterParser(BaseOutputParser):
    """
    Cleans LLM output and formats it into a numbered flow.
    Removes markdown artifacts, extra newlines, quotes,
    and enforces a consistent bullet structure.
    """

    def parse(self, text: str) -> str:
        # 1. Strip quotes and whitespace
        cleaned = text.strip().strip('"').strip("'")

        # 2. Remove repeated blank lines
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        cleaned = "\n".join(lines)

        # 3. Normalize markdown bullets to simple hyphens
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("* ", "- ")
        cleaned = cleaned.replace("•", "- ")

        # 4. Fix numbered lists if LLM added markdown syntax
        cleaned = cleaned.replace("1.", "1)").replace("2.", "2)").replace("3.", "3)")
        cleaned = cleaned.replace("4.", "4)").replace("5.", "5)").replace("6.", "6)")
        cleaned = cleaned.replace("7.", "7)").replace("8.", "8)")
        cleaned = cleaned.replace("9.", "9)").replace("10.", "10)")

        # 5. Ensure double newlines between top-level items
        final_lines = []
        for line in cleaned.split("\n"):
            if line[0:2].isdigit() and ")" in line:
                final_lines.append("\n" + line)
            else:
                final_lines.append(line)
        cleaned = "\n".join(final_lines).lstrip()

        return cleaned

    @property
    def _type(self) -> str:
        return "flow_formatter_parser"
parser = FlowFormatterParser()

prompt = PromptTemplate(
    template="""
      You are a expert coding assistant.
      GO THROUGH THE CODE AND UNDERSTAND IT AND GIVE A DETAILED ANSWER
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | llm | parser


formatted = parser.parse(main_chain.invoke('tell me the schema of the final table'))