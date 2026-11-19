## Prompt 1
Le problème vient de l'ordre d'initialisation : `llmKey` est un champ **static**, mais il n'est rempli que dans le **constructeur**, que `main` n'appelle jamais. Du coup `llmKey` vaut `null` quand tu construis le modèle, d'où le `NullPointerException` (« API key must either be provided... »).

Voici la correction, en chargeant la clé directement dans `main` :

```java
public class RagNaif {

    public static void main(String[] args) {

        Dotenv dotenv = Dotenv.configure()
                .ignoreIfMissing()
                .load();

        // .env d'abord, sinon variable d'environnement système
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

        // ... le reste de ton code inchangé
    }
}
```

Points à vérifier :

1. **Supprime le constructeur et le champ `static`** : ils ne servent à rien ici. Si tu veux les garder, il faudrait faire `new RagNaif()` avant d'utiliser `llmKey`, mais c'est inutilement compliqué.
2. **Emplacement du `.env`** : `Dotenv.load()` cherche le fichier dans le *répertoire de travail*, c'est-à-dire la racine du projet quand tu lances depuis IntelliJ. Pas dans `src/main/resources`.
3. **Nom de la variable** : dans le `.env`, la ligne doit être exactement `GEMINI_KEY=ta_cle` (sans guillemets ni espaces autour du `=`).
4. **Alternative sans `.env`** : la librairie accepte aussi les variables d'environnement `GEMINI_API_KEY` ou `GOOGLE_API_KEY`. Dans IntelliJ : *Run > Edit Configurations > Environment variables*.

Deux autres détails dans ton code :

- Ton package est `com.example.rag` mais la stack trace indique `com.example.belhafiane.rag.RagNaif`. Vérifie que tu lances bien la bonne classe et que le package correspond à l'arborescence des dossiers.
- Vérifie le nom du modèle (`gemini-3.5-flash-lite`) dans la doc Google AI Studio : une erreur sur le nom donnerait une erreur 404 une fois la clé corrigée.

## Prompt 2

Cette erreur vient de ton interface `Assistant` : LangChain4j ne sait pas quel paramètre de `chat(String userMessage)` est le message utilisateur. Avec un seul paramètre, il le déduit normalement grâce aux noms de paramètres du bytecode, mais ça ne fonctionne que si le projet est compilé avec `-parameters`. Ici, le nom est perdu (il devient `arg0`) et l'annotation manque, d'où « does not have a user message defined ».

**Correction** : ajoute `@UserMessage` sur le paramètre.

```java
import dev.langchain4j.service.UserMessage;

interface Assistant {
    String chat(@UserMessage String userMessage);
}
```

C'est la solution la plus simple et la plus robuste, car elle ne dépend d'aucune option de compilation.

**Alternatives** (si tu préfères ne pas annoter) :

- **Maven** : active `-parameters` dans le plugin compiler :
  ```xml
  <plugin>
    <artifactId>maven-compiler-plugin</artifactId>
    <configuration>
      <parameters>true</parameters>
    </configuration>
  </plugin>
  ```
- **IntelliJ** : *Settings > Build, Execution, Deployment > Compiler > Java Compiler > Additional command line parameters* : `-parameters`

**Autre point à surveiller** : ton interface `Assistant` est déclarée **à l'intérieur de `main`** (interface locale). Ça peut poser problème avec les proxys dynamiques selon la version de Java et de LangChain4j. Si l'erreur persiste après le `@UserMessage`, sors-la de la méthode :

```java
public class RagNaif {

    interface Assistant {
        String chat(@UserMessage String userMessage);
    }

    public static void main(String[] args) {
        // ...
    }
}
```

Ton `CompressingQueryTransformer` et ton `RetrievalAugmentor` sont corrects. Le transformer ne fera réellement quelque chose qu'à partir de la deuxième question, quand il y a un historique à condenser avec la nouvelle question. Note aussi qu'il consomme un appel LLM supplémentaire par question, ce qui compte pour ton quota Gemini.