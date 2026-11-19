package com.example.belhafiane.rag;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.onnx.bgesmallenv15.BgeSmallEnV15EmbeddingModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.util.List;
import java.util.Scanner;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;

public class RagNaif {
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
            String chat(String userMessage);
        }

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        "rag.pdf",
                        parser
                );

        System.out.println("Document chargé !");


        DocumentSplitter splitter =
                DocumentSplitters.recursive(1000, 150);

        List<TextSegment> segments =
                splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());


        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();


        Response<List<Embedding>> response =
                embeddingModel.embedAll(segments);

        List<Embedding> embeddings = response.content();

        System.out.println("Embeddings générés : " + embeddings.size());


        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Ingestion terminée !");

        ContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();


        MessageWindowChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);


        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .chatMemory(memory)
                        .contentRetriever(retriever)
                        .build();


        Scanner scanner = new Scanner(System.in);

        System.out.println("\nAssistant RAG prêt !");
        System.out.println("Posez vos questions (tapez 'exit' pour quitter).");

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

    private static void configureLogger() {

        Logger packageLogger = Logger.getLogger("dev.langchain4j");

        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.addHandler(handler);
    }
}
