#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <cstdlib>
#include <locale>
#include <codecvt>
#include <algorithm>

using namespace std;

// Функция для подсчёта количества вхождений подстроки в строку (используется для отладки)
int count_occurrences(const string &str, const string &sub) {
    if (sub.empty()) return 0;
    int count = 0;
    size_t pos = 0;
    while ((pos = str.find(sub, pos)) != string::npos) {
        count++;
        pos++;
    }
    return count;
}

// ---------------------------
// Реализация суффиксного автомата
// ---------------------------

struct SAState {
    int len, link;
    unordered_map<char, int> next;
    int occ; // число конечных позиций, которые попадают в этот класс эквивалентности
};

class SuffixAutomaton {
public:
    vector<SAState> st;
    int last; // индекс состояния, соответствующего всей строке
    
    // Конструктор строит автомат для строки s
    SuffixAutomaton(const string &s) {
        int n = s.size();
        st.resize(2 * n);
        st[0].len = 0;
        st[0].link = -1;
        st[0].occ = 0;
        int sz = 1;
        last = 0;
        for (char c : s) {
            sz = extend(c, sz, last);
        }
        st.resize(sz);
        // Сортируем состояния по убыванию длины
        vector<int> order(sz);
        for (int i = 0; i < sz; i++) order[i] = i;
        sort(order.begin(), order.end(), [&](int a, int b) {
            return st[a].len > st[b].len;
        });
        // Пропагируем значения occ по суффиксным ссылкам:
        for (int i = 0; i < sz; i++) {
            if (st[i].link != -1) {
                st[st[i].link].occ += st[i].occ;
            }
        }
    }
    
    // Расширяет автомат символом c, возвращает новое значение размера автомата
    int extend(char c, int sz, int &last) {
        int cur = sz++;
        st[cur].len = st[last].len + 1;
        st[cur].occ = 1; // новый класс получает occ = 1 (одна конечная позиция)
        int p = last;
        for (; p != -1 && !st[p].next.count(c); p = st[p].link) {
            st[p].next[c] = cur;
        }
        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = sz++;
                st[clone].len = st[p].len + 1;
                st[clone].next = st[q].next; // копируем переходы
                st[clone].link = st[q].link;
                st[clone].occ = 0; // клон изначально не получает собственных вхождений
                for (; p != -1 && st[p].next[c] == q; p = st[p].link) {
                    st[p].next[c] = clone;
                }
                st[q].link = st[cur].link = clone;
            }
        }
        last = cur;
        return sz;
    }
    
    // Возвращает количество вхождений подстроки pattern в корпус (изначальную строку)
    int countOccurrences(const string &pattern) const {
        int cur = 0;
        for (char c : pattern) {
            if (st[cur].next.find(c) == st[cur].next.end())
                return 0;
            cur = st[cur].next.at(c);
        }
        return st[cur].occ;
    }
};

// ---------------------------
// Компоненты BPE-алгоритма
// ---------------------------

// Хэш для пары int
struct IntPairHash {
    size_t operator()(const pair<int, int>& p) const {
        return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
    }
};

// Структура для хранения слова:
// - original: исходное слово из датасета,
// - tokens: вектор токенов (их ID), который изменяется в ходе слияний,
// - pairSet: набор смежных пар токенов для данного слова.
struct WordEntry {
    string original;
    vector<int> tokens;
    unordered_set<pair<int, int>, IntPairHash> pairSet;
};

// Структура для хранения правила слияния.
// frequency – частота нового токена в исходном датасете (вычисляется через суффиксный автомат).
struct MergeRule {
    pair<int, int> merge_pair;
    int new_token;
    int frequency;
};

// Глобальные словари: соответствие токена (строка) ↔ ID.
unordered_map<string, int> token_to_id;
vector<string> id_to_token;

// Возвращает ID для токена; если токена ещё нет, создаёт новый.
int get_token_id(const string &token) {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end())
        return it->second;
    int new_id = id_to_token.size();
    id_to_token.push_back(token);
    token_to_id[token] = new_id;
    return new_id;
}

// Вычисляет множество смежных пар для вектора токенов.
unordered_set<pair<int, int>, IntPairHash> compute_pairs(const vector<int>& tokens) {
    unordered_set<pair<int, int>, IntPairHash> pairs;
    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        pairs.insert({tokens[i], tokens[i+1]});
    }
    return pairs;
}

// Подсчитывает число вхождений пары p в токенизированном слове.
int count_occurrences_in_word(const vector<int>& tokens, const pair<int, int>& p) {
    int count = 0;
    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        if (tokens[i] == p.first && tokens[i+1] == p.second)
            count++;
    }
    return count;
}

// Выполняет слияние указанной пары (target_pair) в векторе токенов: заменяет все вхождения на новый токен.
vector<int> merge_pair_in_word(const vector<int>& tokens, const pair<int, int>& target_pair, int new_token_id) {
    vector<int> new_tokens;
    size_t i = 0;
    while (i < tokens.size()) {
        if (i + 1 < tokens.size() && tokens[i] == target_pair.first && tokens[i+1] == target_pair.second) {
            new_tokens.push_back(new_token_id);
            i += 2;
        } else {
            new_tokens.push_back(tokens[i]);
            i++;
        }
    }
    return new_tokens;
}

// ---------------------------
// Основная программа
// ---------------------------
int main(int argc, char* argv[]) {
    // Устанавливаем локаль для корректной работы с UTF-8
    locale::global(locale("en_US.UTF-8"));
    ios_base::sync_with_stdio(false);
    
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input_file.txt" << endl;
        return EXIT_FAILURE;
    }
    
    string input_filename = argv[1];
    ifstream infile(input_filename);
    if (!infile.is_open()){
        cerr << "Could not open file: " << input_filename << endl;
        return EXIT_FAILURE;
    }
    
    // Имена выходных файлов
    string base_filename = input_filename.substr(0, input_filename.find_last_of('.'));
    string tokens_filename = base_filename + "_tokens.txt";
    string merges_filename = base_filename + "_merges.txt";
    ofstream tokens_file(tokens_filename);
    ofstream merges_file(merges_filename);
    if (!tokens_file.is_open() || !merges_file.is_open()){
        cerr << "Could not create output files" << endl;
        return EXIT_FAILURE;
    }
    
    // Считываем слова
    vector<WordEntry> words;
    wstring_convert<codecvt_utf8<wchar_t>> converter;
    string line;
    int total_words = 0;
    cout << "Reading file..." << endl;
    while (getline(infile, line)) {
        if (line.empty())
            continue;
        istringstream iss(line);
        string word_str;
        // Если строка содержит дополнительные данные (например, tf/df), мы читаем только слово.
        if (!(iss >> word_str)) {
            cerr << "Error reading line: " << line << endl;
            continue;
        }
        WordEntry entry;
        entry.original = word_str;
        // Токенизируем слово: каждый Unicode-символ становится отдельным токеном.
        wstring wide_word = converter.from_bytes(word_str);
        for (auto wc : wide_word) {
            string token = converter.to_bytes(wc);
            int token_id = get_token_id(token);
            entry.tokens.push_back(token_id);
        }
        entry.pairSet = compute_pairs(entry.tokens);
        words.push_back(entry);
        total_words++;
    }
    infile.close();
    cout << "Read " << total_words << " words, initial vocabulary size: " 
         << id_to_token.size() << endl;
    
    // Формируем корпус, объединяя все исходные слова с разделителем (предполагаем, что '#' не встречается в исходном датасете)
    string corpus;
    for (const auto &w : words) {
        corpus += w.original + "#";
    }
    
    // Строим суффиксный автомат для корпуса
    cout << "Building suffix automaton for corpus..." << endl;
    SuffixAutomaton automaton(corpus);
    
    // Глобальная мапа: пара токенов -> множество индексов слов, где пара встречается
    unordered_map<pair<int, int>, unordered_set<int>, IntPairHash> pair_to_word_indices;
    for (size_t i = 0; i < words.size(); i++) {
        for (const auto &p : words[i].pairSet) {
            pair_to_word_indices[p].insert(i);
        }
    }
    
    // Вектор для хранения правил слияния
    vector<MergeRule> merges;
    cout << "Starting BPE merging process..." << endl;
    int iteration = 0;
    while (true) {
        iteration++;
        // Находим лучшую пару (с максимальной суммарной частотой) по текущей токенизации
        pair<int, int> best_pair = {-1, -1};
        int best_freq = 0;
        for (const auto &kv : pair_to_word_indices) {
            const auto &p = kv.first;
            int freq = 0;
            for (int idx : kv.second) {
                freq += count_occurrences_in_word(words[idx].tokens, p);
            }
            if (freq > best_freq) {
                best_freq = freq;
                best_pair = p;
            }
        }
        if (best_freq < 2)
            break; // если нет пар с частотой >= 2, завершаем слияния
        
        // Создаём новый токен как конкатенация строковых представлений сливаемых токенов
        string new_token_str = id_to_token[best_pair.first] + id_to_token[best_pair.second];
        int new_token_id = get_token_id(new_token_str);
        
        // Подсчитываем частоту нового токена в исходном датасете через суффиксный автомат
        int original_freq = automaton.countOccurrences(new_token_str);
        merges.push_back({best_pair, new_token_id, original_freq});
        cout << "Iteration " << iteration << ": merging ("
             << id_to_token[best_pair.first] << ", " << id_to_token[best_pair.second]
             << ") -> " << new_token_str << ", original frequency: " << original_freq << endl;
        
        // Получаем индексы слов, где встречается выбранная пара
        unordered_set<int> affected_words = pair_to_word_indices[best_pair];
        // Обновляем токенизацию только для этих слов
        for (int idx : affected_words) {
            WordEntry &word = words[idx];
            auto old_pairSet = word.pairSet;
            vector<int> new_tokens = merge_pair_in_word(word.tokens, best_pair, new_token_id);
            word.tokens = new_tokens;
            unordered_set<pair<int, int>, IntPairHash> new_pairSet = compute_pairs(word.tokens);
            // Для пар, которые исчезли, удаляем индекс слова из глобальной мапы
            for (const auto &p : old_pairSet) {
                if (new_pairSet.find(p) == new_pairSet.end()) {
                    auto it = pair_to_word_indices.find(p);
                    if (it != pair_to_word_indices.end()) {
                        it->second.erase(idx);
                        if (it->second.empty())
                            pair_to_word_indices.erase(it);
                    }
                }
            }
            // Для новых пар, которые появились, добавляем индекс слова
            for (const auto &p : new_pairSet) {
                if (old_pairSet.find(p) == old_pairSet.end()) {
                    pair_to_word_indices[p].insert(idx);
                }
            }
            word.pairSet = new_pairSet;
        }
        // Удаляем обработанную пару из глобальной мапы
        pair_to_word_indices.erase(best_pair);
    }
    cout << "Merging process completed after " << iteration - 1 << " iterations." << endl;
    
    // Подсчитываем финальные частоты для каждого токена по исходному датасету через суффиксный автомат
    unordered_map<int, int> final_frequencies;
    for (size_t token_id = 0; token_id < id_to_token.size(); token_id++) {
        int freq = automaton.countOccurrences(id_to_token[token_id]);
        if (freq > 0)
            final_frequencies[token_id] = freq;
    }
    
    cout << "Final vocabulary size: " << final_frequencies.size() << endl;
    
    // Вывод финальных токенов с их частотами
    tokens_file << "Token\tFrequency\n";
    for (const auto &kv : final_frequencies) {
        tokens_file << id_to_token[kv.first] << "\t" << kv.second << "\n";
    }
    
    // Вывод правил слияния
    merges_file << "Merge Rules (with original frequencies):\n";
    for (const auto &rule : merges) {
        merges_file << "(" << id_to_token[rule.merge_pair.first] << ", "
                    << id_to_token[rule.merge_pair.second] << ") -> "
                    << id_to_token[rule.new_token] << ", frequency: "
                    << rule.frequency << "\n";
    }
    
    tokens_file.close();
    merges_file.close();
    cout << "Tokens written to: " << tokens_filename << endl;
    cout << "Merge rules written to: " << merges_filename << endl;
    
    return EXIT_SUCCESS;
}
