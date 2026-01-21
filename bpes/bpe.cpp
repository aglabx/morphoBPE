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

// Для удобства
using std::string;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

// Функция для подсчёта количества вхождений подстроки sub в строку str (с перекрытиями)
int count_occurrences(const string& str, const string& sub) {
    if (sub.empty()) return 0;
    int count = 0;
    size_t pos = 0;
    while ((pos = str.find(sub, pos)) != string::npos) {
        count++;
        pos += 1; // сдвиг на 1 для поиска перекрывающихся вхождений
    }
    return count;
}

// Хэш-функция для пары int
struct IntPairHash {
    std::size_t operator()(const pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Структура для хранения слова:
// - original: исходное слово (строка) из датасета,
// - tokens: вектор токенов (ID), который будет изменяться в ходе слияний,
// - pairSet: набор смежных пар токенов для этого слова.
struct WordEntry {
    string original; 
    vector<int> tokens;
    unordered_set<pair<int, int>, IntPairHash> pairSet;
};

// Структура для хранения правила слияния.
// frequency – частота нового токена в исходном датасете (подсчитанная по original).
struct MergeRule {
    pair<int, int> merge_pair; 
    int new_token;                  
    int frequency;                  
};

// Глобальные словари: соответствие токена (строка) <-> ID.
unordered_map<string, int> token_to_id;
vector<string> id_to_token;

// Возвращает ID для токена. Если токена ещё нет – добавляет его.
int get_token_id(const string& token) {
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

// Подсчитывает число вхождений пары p в векторе токенов (как смежных элементов).
int count_occurrences_in_word(const vector<int>& tokens, const pair<int, int>& p) {
    int count = 0;
    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        if (tokens[i] == p.first && tokens[i+1] == p.second)
            count++;
    }
    return count;
}

// Выполняет слияние указанной пары (target_pair) в векторе токенов:
// заменяет все вхождения пары на новый токен (new_token_id).
vector<int> merge_pair_in_word(const vector<int>& tokens, const pair<int, int>& target_pair, int new_token_id) {
    vector<int> new_tokens;
    size_t i = 0;
    while (i < tokens.size()) {
        if (i < tokens.size()-1 && tokens[i] == target_pair.first && tokens[i+1] == target_pair.second) {
            new_tokens.push_back(new_token_id);
            i += 2;
        } else {
            new_tokens.push_back(tokens[i]);
            i++;
        }
    }
    return new_tokens;
}

int main(int argc, char* argv[]) {
    // Устанавливаем локаль для корректной работы с UTF-8
    std::locale::global(std::locale("en_US.UTF-8"));
    std::ios_base::sync_with_stdio(false);
    
    if (argc < 2) {
        cerr << "Использование: " << argv[0] << " input_file.txt" << endl;
        return EXIT_FAILURE;
    }
    
    string input_filename = argv[1];
    std::ifstream infile(input_filename);
    if (!infile.is_open()){
        cerr << "Не удалось открыть файл: " << input_filename << endl;
        return EXIT_FAILURE;
    }
    
    // Создаем имена выходных файлов.
    string base_filename = input_filename.substr(0, input_filename.find_last_of('.'));
    string tokens_filename = base_filename + "_tokens.txt";
    string merges_filename = base_filename + "_merges.txt";
    std::ofstream tokens_file(tokens_filename);
    std::ofstream merges_file(merges_filename);
    if (!tokens_file.is_open() || !merges_file.is_open()){
        cerr << "Не удалось создать выходные файлы" << endl;
        return EXIT_FAILURE;
    }
    
    vector<WordEntry> words;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    string line;
    int total_words = 0;
    cout << "Читаем файл..." << endl;
    while (getline(infile, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        string word_str;
        // Если в строке присутствуют tf/df, они игнорируются — считываем только слово.
        if (!(iss >> word_str)) {
            cerr << "Ошибка чтения строки: " << line << endl;
            continue;
        }
        WordEntry entry;
        entry.original = word_str;  // сохраняем исходное слово
        
        // Токенизируем слово: каждый Unicode-символ становится отдельным токеном.
        std::wstring wide_word = converter.from_bytes(word_str);
        for (const auto &wc : wide_word) {
            string token = converter.to_bytes(wc);
            int token_id = get_token_id(token);
            entry.tokens.push_back(token_id);
        }
        entry.pairSet = compute_pairs(entry.tokens);
        words.push_back(entry);
        total_words++;
    }
    infile.close();
    cout << "Прочитано " << total_words << " слов, начальный размер словаря: " 
         << id_to_token.size() << endl;
    
    // Глобальная мапа: пара токенов -> множество индексов слов, где пара встречается.
    unordered_map<pair<int, int>, unordered_set<int>, IntPairHash> pair_to_word_indices;
    for (size_t i = 0; i < words.size(); i++) {
        for (const auto &p : words[i].pairSet) {
            pair_to_word_indices[p].insert(i);
        }
    }
    
    vector<MergeRule> merges;
    cout << "Начинаем процесс слияний..." << endl;
    int iteration = 0;
    while (true) {
        iteration++;
        // Находим лучшую пару (с максимальной суммарной частотой) по текущей токенизации.
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
            break; // если пар с частотой >= 2 нет, завершаем слияния
        
        // Создаем новый токен как конкатенацию строковых представлений сливаемых токенов.
        string new_token_str = id_to_token[best_pair.first] + id_to_token[best_pair.second];
        int new_token_id = get_token_id(new_token_str);
        
        // Подсчитываем частоту нового токена в исходном датасете (сканируем оригинальные слова).
        int original_freq = 0;
        for (const auto &w : words) {
            original_freq += count_occurrences(w.original, new_token_str);
        }
        merges.push_back({best_pair, new_token_id, original_freq});
        
        cout << "Итерация " << iteration << ": слияние (" 
             << id_to_token[best_pair.first] << ", " << id_to_token[best_pair.second] 
             << ") -> " << new_token_str << ", частота в исходном датасете: " 
             << original_freq << endl;
        
        // Получаем индексы слов, где встречается выбранная пара.
        unordered_set<int> affected_words = pair_to_word_indices[best_pair];
        // Обновляем токенизацию только для этих слов.
        for (int idx : affected_words) {
            WordEntry &word = words[idx];
            auto old_pairSet = word.pairSet; // запоминаем старый набор пар
            vector<int> new_tokens = merge_pair_in_word(word.tokens, best_pair, new_token_id);
            word.tokens = new_tokens;
            unordered_set<pair<int, int>, IntPairHash> new_pairSet = compute_pairs(word.tokens);
            // Удаляем из глобальной мапы пары, которые исчезли.
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
            // Добавляем новые пары.
            for (const auto &p : new_pairSet) {
                if (old_pairSet.find(p) == old_pairSet.end()) {
                    pair_to_word_indices[p].insert(idx);
                }
            }
            word.pairSet = new_pairSet;
        }
        // Удаляем обработанную пару.
        pair_to_word_indices.erase(best_pair);
    }
    cout << "Процесс слияний завершён (итераций: " << iteration - 1 << ")." << endl;
    
    // Подсчитываем финальные частоты для каждого токена по исходным словам.
    // Для каждого токена (из всей итоговой лексики) суммируем количество вхождений в исходном датасете.
    unordered_map<int, int> final_frequencies;
    for (size_t token_id = 0; token_id < id_to_token.size(); token_id++) {
        int freq = 0;
        const string &token_str = id_to_token[token_id];
        for (const auto &w : words) {
            freq += count_occurrences(w.original, token_str);
        }
        if (freq > 0)
            final_frequencies[token_id] = freq;
    }
    
    cout << "Размер итогового словаря (токенов): " << final_frequencies.size() << endl;
    
    // Вывод финальных токенов и их частот (подсчитанных в исходных словах)
    tokens_file << "Token\tFrequency\n";
    for (const auto &kv : final_frequencies) {
        tokens_file << id_to_token[kv.first] << "\t" << kv.second << "\n";
    }
    
    // Вывод правил слияния (с указанием частоты нового токена в исходном датасете)
    merges_file << "Merge rules (with original frequencies):\n";
    for (const auto &rule : merges) {
        merges_file << "(" << id_to_token[rule.merge_pair.first] << ", " 
                    << id_to_token[rule.merge_pair.second] << ") -> " 
                    << id_to_token[rule.new_token] << ", frequency: " << rule.frequency << "\n";
    }
    
    tokens_file.close();
    merges_file.close();
    cout << "Токены записаны в файл: " << tokens_filename << endl;
    cout << "Правила слияния записаны в файл: " << merges_filename << endl;
    
    return EXIT_SUCCESS;
}
