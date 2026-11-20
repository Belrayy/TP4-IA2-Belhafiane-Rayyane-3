package com.example.belhafiane.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;

import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;

import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;
import io.github.cdimascio.dotenv.Dotenv;

import java.util.List;
import java.util.Scanner;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;


public class Test4 {
    private static void configureLogger() {
        Logger logger = Logger.getLogger("dev.langchain4j");
        logger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        logger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();

        Dotenv dotenv = Dotenv.configure()
                .ignoreIfMissing()
                .load();

        String llmKey = dotenv.get("GEMINI_KEY");
        String tavilyKey = dotenv.get("TAVILY_API_KEY");

        if (llmKey == null || llmKey.isBlank()) {
            throw new IllegalStateException(
                    "GEMINI_KEY introuvable : vérifie ton fichier .env à la racine du projet");
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.7)
                .build();

        interface Assistant {
            // Prend un message de l'utilisateur et retourne une réponse du LLM.
            String chat(@UserMessage String userMessage);
        }

        // 2. Chargement du document PDF
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", parser);

        DocumentSplitter splitter = DocumentSplitters.recursive(1000, 150);
        List<TextSegment> segments = splitter.split(document);

        EmbeddingModel embeddingModel = new BgeSmallEnV15EmbeddingModel();
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        ContentRetriever pdfRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)
                .build();

        QueryRouter queryRouter = new DefaultQueryRouter(
                pdfRetriever,
                webRetriever
        );

        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        Scanner scanner = new Scanner(System.in);

        System.out.println("Assistant RAG avec recherche Web prêt !");
        System.out.println("Tapez 'exit' pour quitter.");

        while (true) {
            System.out.print("\nVous : ");
            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                break;
            }

            String response = assistant.chat(question);
            System.out.println("\nAssistant : " + response);
        }

        scanner.close();
    }
}
