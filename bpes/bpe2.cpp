#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

// -----------------------
// Структура для узла двусвязного списка
// -----------------------
struct Node {
    int token;      // id токена
    Node* prev;
    Node* next;
    bool removed;   // для пометки удалённых узлов
    Node(int t) : token(t), prev(nullptr), next(nullptr), removed(false) {}
};

// Глобальные указатели на голову и хвост списка
Node* head = nullptr;
Node* tail = nullptr;

// Функция для добавления узла в конец списка
void appendNode(Node* node) {
    if (!head) {
        head = tail = node;
    } else {
        tail->next = node;
        node->prev = tail;
        tail = node;
    }
}

// -----------------------
// Словарь токенов: строка → id и вектор id → строка
// -----------------------
unordered_map<string, int> token_to_id;
vector<string> id_to_token;

// Функция получения id для токена; если токена нет – добавляет его.
// Токен 0 зарезервирован для спейсера ("<spacer>").
int getTokenId(const string &token) {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end())
        return it->second;
    int id = id_to_token.size();
    id_to_token.push_back(token);
    token_to_id[token] = id;
    return id;
}

// -----------------------
// Структуры для хранения пар токенов
// -----------------------
struct PairHash {
    size_t operator()(const pair<int,int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

struct PairData {
    int freq;              // число вхождений
    vector<Node*> occ;     // список указателей на первую ноду пары
};

typedef unordered_map<pair<int,int>, PairData, PairHash> PairMap;

// Добавление вхождения для пары p (указатель на первую ноду)
void addOccurrence(PairMap &pmap, const pair<int,int>& p, Node* node) {
    pmap[p].occ.push_back(node);
    pmap[p].freq = pmap[p].occ.size();
}

// Удаление указанного указателя из вхождений для пары p
void removeOccurrence(PairMap &pmap, const pair<int,int>& p, Node* node) {
    if (pmap.find(p) != pmap.end()) {
        auto &vec = pmap[p].occ;
        vec.erase(remove(vec.begin(), vec.end(), node), vec.end());
        pmap[p].freq = vec.size();
        if (vec.empty())
            pmap.erase(p);
    }
}

// Построение мапы пар по всему списку
void buildPairMap(PairMap &pmap) {
    pmap.clear();
    Node* cur = head;
    while (cur && cur->next) {
        // Игнорируем пары, если хотя бы один токен – спейсер (id 0)
        if (cur->token != 0 && cur->next->token != 0) {
            pair<int,int> p = {cur->token, cur->next->token};
            addOccurrence(pmap, p, cur);
        }
        cur = cur->next;
    }
}

// -----------------------
// Основная программа
// -----------------------
int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.txt\n";
        return 1;
    }
    
    // Зарезервируем токен 0 для спейсера
    token_to_id["<spacer>"] = 0;
    id_to_token.push_back("<spacer>");

    // Чтение слов из файла и построение двусвязного списка.
    // Каждая строка – слово; токенизация: каждый символ становится отдельным токеном.
    ifstream infile(argv[1]);
    string line;
    bool firstWord = true;
    while(getline(infile, line)) {
        if(line.empty())
            continue;
        // Для каждого символа слова
        for (char c : line) {
            string s(1, c);
            int tid = getTokenId(s);
            Node* node = new Node(tid);
            appendNode(node);
        }
        // После слова вставляем спейсер
        Node* spacer = new Node(0); // 0 – спейсер
        appendNode(spacer);
    }
    infile.close();
    // Удалим последний спейсер, если он есть
    if (tail && tail->token == 0) {
        Node* tmp = tail;
        tail = tail->prev;
        if (tail) tail->next = nullptr;
        else head = nullptr;
        delete tmp;
    }
    
    // Построим глобальную мапу пар
    PairMap pairMap;
    buildPairMap(pairMap);
    
    // Основной цикл слияний BPE:
    // Пока есть пара, встречающаяся хотя бы 2 раза, выбираем самую частую.
    while(true) {
        pair<int,int> bestPair = {-1, -1};
        int bestFreq = 0;
        // Поиск пары с максимальной частотой
        for (auto &entry : pairMap) {
            if (entry.second.freq > bestFreq) {
                bestFreq = entry.second.freq;
                bestPair = entry.first;
            }
        }
        if (bestFreq < 2)
            break;
        
        // Создаём новый токен как конкатенация строковых представлений сливаемых токенов
        string newTokenStr = id_to_token[bestPair.first] + id_to_token[bestPair.second];
        int newTokenId = getTokenId(newTokenStr);
        
        // Вывод информации о слиянии (можно убрать)
        cout << "Merging pair (" << id_to_token[bestPair.first] << ", " 
             << id_to_token[bestPair.second] << ") -> " << newTokenStr 
             << " [freq=" << bestFreq << "]\n";
        
        // Получаем копию списка вхождений для выбранной пары
        vector<Node*> occList = pairMap[bestPair].occ;
        
        // Обрабатываем каждое вхождение
        for (Node* left : occList) {
            if (!left || left->removed || !left->next)
                continue;
            Node* right = left->next;
            if (right->removed)
                continue;
            // Проверяем, что пара действительно соответствует
            if (left->token != bestPair.first || right->token != bestPair.second)
                continue;
            
            // Слияние: в узле left заменяем токен на новый, а узел right удаляем
            left->token = newTokenId;
            Node* after = right->next;
            left->next = after;
            if (after)
                after->prev = left;
            right->removed = true; // помечаем как удалённый
            
            // Обновляем мапу пар для смежных пар:
            // 1. Пара, образуемая между left->prev и left
            if (left->prev && !left->prev->removed && left->prev->token != 0 && left->token != 0) {
                pair<int,int> oldPair = {left->prev->token, bestPair.first}; // до слияния left имел bestPair.first
                removeOccurrence(pairMap, oldPair, left->prev);
                pair<int,int> newPair = {left->prev->token, newTokenId};
                addOccurrence(pairMap, newPair, left->prev);
            }
            // 2. Пара, образуемая между left и left->next
            if (left->next && !left->next->removed && left->token != 0 && left->next->token != 0) {
                // Раньше в паре участвовал right (с токеном bestPair.second)
                pair<int,int> oldPair = {bestPair.second, left->next->token};
                removeOccurrence(pairMap, oldPair, right); // right был началом такой пары
                pair<int,int> newPair = {newTokenId, left->next->token};
                addOccurrence(pairMap, newPair, left);
            }
        }
        // После обработки выбранной пары удаляем её запись из мапы
        pairMap.erase(bestPair);
    }
    
    // Вывод финального результата: проходим по списку и выводим токены
    cout << "\nFinal token sequence:\n";
    Node* cur = head;
    while(cur) {
        // Игнорируем спейсер (можно выводить пробел)
        if (cur->token == 0)
            cout << " ";
        else
            cout << id_to_token[cur->token];
        cur = cur->next;
    }
    cout << "\n";
    
    // Очистка памяти: удаляем все узлы списка
    cur = head;
    while(cur) {
        Node* nxt = cur->next;
        delete cur;
        cur = nxt;
    }
    
    return 0;
}
