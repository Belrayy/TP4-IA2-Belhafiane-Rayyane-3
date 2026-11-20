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
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import io.github.cdimascio.dotenv.Dotenv;

import java.util.*;

import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;

import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;

public class Test3 {
    private static void configureLogger() {

        Logger logger =
                Logger.getLogger("dev.langchain4j");

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

        ApacheTikaDocumentParser parser =
                new ApacheTikaDocumentParser();

        Document document =
                ClassPathDocumentLoader.loadDocument(
                        "rag.pdf", parser
                );

        DocumentSplitter splitter =
                DocumentSplitters.recursive(1000, 150);

        List<TextSegment> segments =
                splitter.split(document);

        EmbeddingModel embeddingModel =
                new BgeSmallEnV15EmbeddingModel();

        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);



        ContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();


        PromptTemplate promptTemplate =
                PromptTemplate.from(
                        "Est-ce que la requête '{{query}}' porte sur l'IA ? " +
                                "Réponds seulement par 'oui', 'non' ou 'peut-être'."
                );


        QueryRouter queryRouter = new QueryRouter() {

            @Override
            public Collection<ContentRetriever> route(Query query) {

                String prompt = promptTemplate
                        .apply(Map.of("query", query.text()))
                        .text();

                System.out.println("\n--- ROUTAGE ---");
                System.out.println("Prompt envoyé au LM :");
                System.out.println(prompt);

                String decision = model.chat(prompt);

                System.out.println("Décision du LM : " + decision);

                String normalizedDecision = decision
                        .trim()
                        .toLowerCase(Locale.ROOT)
                        .replaceAll("[.!?\\s]+$", "");

                if (normalizedDecision.equals("non")) {

                    System.out.println("RAG désactivé.");

                    return List.of();
                }

                System.out.println("RAG activé.");

                return List.of(retriever);
            }
        };


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



        System.out.println("\nAssistant RAG conditionnel prêt !");
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
