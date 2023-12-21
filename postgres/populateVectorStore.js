const { PDFLoader } = require('langchain/document_loaders/fs/pdf');
const { VercelPostgres } = require("langchain/vectorstores/vercel_postgres")
const { OpenAIEmbeddings } = require("langchain/embeddings/openai")
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter")

const populateVectorStore = async () => {
    console.log('Loading PDF...')

    const loader = new PDFLoader("../intex-spa.pdf");

    const docs = await loader.load();

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    })

    const splitDocs = await splitter.splitDocuments(docs);

    const vercelPostgresStore = await VercelPostgres.initialize(
        new OpenAIEmbeddings()
    );

    console.log('Uploading PDF to PostGres...')

    // Load the docs into the vector store
    try {
        await vercelPostgresStore.addDocuments(splitDocs)
        console.log('Docs uploaded successfully')
    }
    catch(e) {
        console.log('Unable to upload docs to PostGres' + JSON.stringify(e))
    }
    finally {
        process.exit()
    }
};

populateVectorStore();