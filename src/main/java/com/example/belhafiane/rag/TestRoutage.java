package com.example.belhafiane.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import io.github.cdimascio.dotenv.Dotenv;

import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Scanner;

public class TestRoutage{
    private static EmbeddingStore<TextSegment> createEmbeddingStore(
            String fileName,
            EmbeddingModel embeddingModel) {

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument(fileName, parser);

        DocumentSplitter splitter =
                DocumentSplitters.recursive(1000, 150);

        List<TextSegment> segments = splitter.split(document);

        System.out.println(fileName + " : " +
                segments.size() + " segments");

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Ingestion terminée : " + fileName);

        return embeddingStore;
    }

    private static ContentRetriever createRetriever(
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel,
            String name) {

        return EmbeddingStoreContentRetriever.builder()
                .displayName(name)
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    private static void configureLogger() {

        java.util.logging.Logger logger =
                java.util.logging.Logger.getLogger("dev.langchain4j");

        logger.setLevel(java.util.logging.Level.FINE);

        java.util.logging.ConsoleHandler handler =
                new java.util.logging.ConsoleHandler();

        handler.setLevel(java.util.logging.Level.FINE);

        logger.addHandler(handler);
    }

    public static void main(String[] args) {

        configureLogger();

        Dotenv dotenv = Dotenv.configure()
                .ignoreIfMissing()
                .load();

        String llmKey = dotenv.get("GEMINI_KEY");
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

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        EmbeddingStore<TextSegment> storeIA =
                createEmbeddingStore("rag.pdf", embeddingModel);

        EmbeddingStore<TextSegment> storeReseaux =
                createEmbeddingStore("reseaux.pdf", embeddingModel);


        ContentRetriever retrieverIA =
                createRetriever(storeIA, embeddingModel, "Documents IA");

        ContentRetriever retrieverReseaux =
                createRetriever(storeReseaux, embeddingModel, "Documents Réseaux");


        Map<ContentRetriever, String> retrieverToDescription =
                new LinkedHashMap<>();

        retrieverToDescription.put(
                retrieverIA,
                "Documents concernant l'intelligence artificielle, " +
                        "le RAG, le fine-tuning et les modèles de langage."
        );

        retrieverToDescription.put(
                retrieverReseaux,
                "Documents concernant les réseaux informatiques, " +
                        "les adresses IP, les protocoles TCP/IP et les architectures réseau."
        );


        QueryRouter queryRouter =
                new LanguageModelQueryRouter(model, retrieverToDescription);


        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();


        MessageWindowChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);


        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(memory)
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();


        Scanner scanner = new Scanner(System.in);

        System.out.println("Assistant RAG avec routage prêt !");
        System.out.println("Tapez 'exit' pour quitter.");

        while (true) {

            System.out.print("\nVous : ");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("exit")) {
                break;
            }

            String reponse = assistant.chat(question);

            System.out.println("\nAssistant : " + reponse);
        }

        scanner.close();
    }
}
