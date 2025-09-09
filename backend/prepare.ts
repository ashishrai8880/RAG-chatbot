import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Pinecone as PinecodeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: "text-embedding-3-small",
});

const pinecone = new PinecodeClient({ apiKey: process.env.PINECONE_API_KEY });

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

export const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
  maxConcurrency: 5,
});

export async function indexTheDocument(filePath: string) {
  console.log("checking api key : ", { apiKey: process.env.PINECONE_API_KEY });
  const loader = new PDFLoader(filePath, { splitPages: false });
  const doc = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  // @ts-ignore
  const texts = await textSplitter.splitText(doc[0].pageContent);

  const documents = texts.map((chunk: any) => {
    return {
      pageContent: chunk,
      // @ts-ignore
      metadata: doc[0].metadata,
    };
  });

  await vectorStore.addDocuments(documents);
}
