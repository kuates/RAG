
# 🧠 RAG Demo: Chatbot avec Mémoire et Vectorisation FAISS

Ce projet est un **démonstrateur minimaliste** d’un système **Retrieval-Augmented Generation (RAG)** construit avec des outils modernes tels que **LangChain**, **OpenAI**, **FAISS**, et le **Text Splitter** `RecursiveCharacterTextSplitter`.

---

## 🎯 Objectif

Permettre à un utilisateur d'interroger un corpus local (PDF ou TXT) avec une interface simple en ligne de commande, où les réponses sont générées par un LLM **en utilisant uniquement les documents fournis** comme source d'information.

Le système conserve **l'historique de conversation**, ce qui permet des échanges contextuels successifs.

---

## 🛠️ Fonctionnalités Clés

- ✅ Lecture de fichiers `.txt` et `.pdf`
- ✅ Chunking intelligent via `RecursiveCharacterTextSplitter`
- ✅ Vectorisation via `OpenAI Embeddings` (`text-embedding-3-small`)
- ✅ Indexation rapide via `FAISS`
- ✅ Recherche sémantique des documents
- ✅ Réponses générées par `gpt-4o-mini` **avec contexte**
- ✅ Conversation avec **mémoire de chat**

---

## 🧱 Architecture du Pipeline

```text
                  ┌────────────────────┐
                  │ Fichiers PDF / TXT │
                  └────────┬───────────┘
                           ▼
         ┌────────────────────────────────────┐
         │ Extraction de texte + découpages   │
         │ (RecursiveCharacterTextSplitter)   │
         └────────┬───────────────────────────┘
                  ▼
         ┌──────────────────────┐
         │ Embedding OpenAI     │
         │ (text-embedding-3)   │
         └────────┬─────────────┘
                  ▼
         ┌────────────────────┐
         │  FAISS Vector Store │
         └────────┬───────────┘
                  ▼
         ┌─────────────────────┐
         │ Recherche Sémantique│
         └────────┬────────────┘
                  ▼
         ┌────────────────────────────────┐
         │ LLM OpenAI (gpt-4o-mini)       │
         │ avec prompt basé sur le contexte│
         └────────────────────────────────┘
```

---

## 📂 Arborescence minimale

```bash
project/
├── rag_demo_splitter_history.py
├── .env                       # Contient votre clé API OpenAI
├── faiss.index                # Stockage FAISS (auto-généré)
├── faiss.index.meta.json      # Métadonnées des chunks
└── RAG_DATA/                  # Vos fichiers PDF / TXT
```

---

## ⚙️ Installation

```bash
pip install openai==1.* faiss-cpu numpy pypdf langchain tqdm python-dotenv
```

---

## 📄 Configuration

Créez un fichier `.env` avec votre clé OpenAI :

```env
openai_key=sk-...
```

---

## ▶️ Utilisation

Placez vos documents `.pdf` ou `.txt` dans le dossier `RAG_DATA/`.

Lancez le script :

```bash
python rag_demo_splitter_history.py
```

Posez vos questions directement :

```text
💬  Ask (Ctrl-C to quit): Quel est le résumé du document sur la stratégie IA ?
```

Le modèle affichera :

- Le contexte retrouvé (extraits pertinents)
- La réponse générée par l’IA
- Un historique des échanges est maintenu tout au long de la session

---

## 🧠 Notes Techniques

- **Historique de chat** : conservé côté local uniquement pendant la session
- **LangChain** est utilisé uniquement pour le `text_splitter`, dans une logique de clarté et de découplage
- Aucun framework web, pas de base de données — tout est simple et local
- Index FAISS persistant entre les exécutions

---

# 🧠 RAG Demo — Manual vs LangChain

Ce projet propose deux démonstrations complètes de Retrieval-Augmented Generation (RAG) en Python :  
une approche **manuelle avec FAISS + OpenAI**, et une autre **modulaire via le framework LangChain**.

---

## 1. ✨ Fonctionnalités

- Lecture de fichiers `.txt` et `.pdf`
- Découpage de texte (`RecursiveCharacterTextSplitter`)
- Indexation vectorielle via **FAISS**
- Embedding avec OpenAI (`text-embedding-3-small`)
- Chat avec **GPT-4o-mini**
- **Historique de conversation** dans les deux cas
- Deux implémentations :
  - Bas niveau (sans framework)
  - LangChain avec mémoire et chaines préconstruites

---

## 2. 📦 Prérequis

- Compte OpenAI et clé API active
- Python 3.9+
- Fichiers `.txt` et `.pdf` dans le dossier `RAG_DATA`

---

## 3. ⚙️ Installation

### A. Version manuelle

```bash
pip install openai==1.* faiss-cpu numpy pypdf langchain tqdm python-dotenv
```

### B. Version LangChain

```bash
pip install langchain-openai langchain-community faiss-cpu pypdf tqdm python-dotenv
```

---

## 4. 🚀 Démarrage rapide

1. Placez vos fichiers `.pdf` ou `.txt` dans le dossier `RAG_DATA`
2. Créez un fichier `.env` avec :
```
openai_key=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
3. Lancez l'une des deux versions :

### ➤ Script 1 : Approche manuelle
```bash
python rag_demo_splitter_history.py
```

### ➤ Script 2 : Version LangChain
```bash
python rag_langchain_chat_history.py
```

---

## 5. 🧪 Implémentation manuelle (`rag_demo_splitter_history.py`)

Architecture minimaliste construite sans framework :
- Découpage manuel des documents
- Embedding via `openai.embeddings.create()`
- Indexation FAISS directe
- Recherche vectorielle manuelle
- Contexte injecté manuellement dans le prompt
- Historique géré avec une liste Python

**Avantages** : Contrôle total, pédagogie.  
**Inconvénients** : Plus verbeux, moins flexible.

---

## 6. 🧩 Implémentation LangChain (`rag_langchain_chat_history.py`)

Basée sur les abstractions de LangChain :

- **Chargement des documents** : `TextLoader`, `PyPDFLoader`
- **Découpage** : `RecursiveCharacterTextSplitter`
- **Embedding** : `OpenAIEmbeddings`
- **Indexation** : `FAISS` via `langchain_community.vectorstores`
- **Chaîne conversationnelle** : `ConversationalRetrievalChain`
- **Mémoire** : `ConversationBufferMemory` (persistance de l'historique)
- **Prompt personnalisé** : via `ChatPromptTemplate`

Cette version automatise la création des messages système, la gestion de l'historique, et la concaténation du contexte.

**Avantages** : Structuré, maintenable, extensible.  
**Inconvénients** : Moins de visibilité sur les appels internes.

---

## 7. 🔍 Comparaison des approches

| Critère              | Version manuelle                   | Version LangChain                       |
|----------------------|------------------------------------|-----------------------------------------|
| Contrôle             | 🟢 Complet                         | 🔸 Abstrait                             |
| Complexité code      | 🔴 Plus verbeux                    | 🟢 Plus concis                          |
| Transparence         | 🟢 Maximale                        | 🔸 Moins lisible                        |
| Modularité           | 🔸 Moins flexible                   | 🟢 Très modulaire                       |
| Maintenance          | 🔸 Plus coûteuse                   | 🟢 Facile grâce aux composants LangChain|
| Usage en production  | 🔸 Requiert du refactoring         | 🟢 Prêt pour industrialisation          |

---

## 8. ⚠️ Limitations

- Pas de gestion fine des métadonnées documentaires
- Pas de UI graphique (interface CLI uniquement)
- Les modèles OpenAI nécessitent une connexion internet et une API Key valide

---

## 10. 📚 Références

- [LangChain](https://python.langchain.com)
- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

---

