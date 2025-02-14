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

// Хэш для пары целых чисел
struct IntPairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 1);
    }
};

// Структура для хранения слова: вектор токенов (ID), а также набор смежных пар и TF/DF (если понадобится)
struct WordEntry {
    std::vector<int> tokens;
    std::unordered_set<std::pair<int, int>, IntPairHash> pairSet;
    int tf;
    int df;
};

// Структура для хранения правила слияния
struct MergeRule {
    std::pair<int, int> merge_pair; // пара токенов, которую слили
    int new_token;                  // ID нового токена (конкатенация)
    int frequency;                  // суммарная частота (учитывая tf слов)
};

// Глобальные мапы для соответствия токен (строка) <-> ID
std::unordered_map<std::string, int> token_to_id;
std::vector<std::string> id_to_token;

// Функция для получения ID токена; если токена ещё нет, то создаём его
int get_token_id(const std::string& token) {
    auto it = token_to_id.find(token);
    if (it != token_to_id.end()) {
        return it->second;
    } else {
        int new_id = id_to_token.size();
        id_to_token.push_back(token);
        token_to_id[token] = new_id;
        return new_id;
    }
}

// Функция, вычисляющая множество смежных пар для вектора токенов (каждая пара хранится только 1 раз)
std::unordered_set<std::pair<int, int>, IntPairHash> compute_pairs(const std::vector<int>& tokens) {
    std::unordered_set<std::pair<int, int>, IntPairHash> pairs;
    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        pairs.insert({tokens[i], tokens[i+1]});
    }
    return pairs;
}

// Подсчитывает количество вхождений пары p в слово (как смежных токенов)
int count_occurrences_in_word(const std::vector<int>& tokens, const std::pair<int, int>& p) {
    int count = 0;
    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        if (tokens[i] == p.first && tokens[i+1] == p.second)
            count++;
    }
    return count;
}

// Функция для слияния указанной пары в одном слове
std::vector<int> merge_pair_in_word(const std::vector<int>& tokens, const std::pair<int, int>& target_pair, int new_token_id) {
    std::vector<int> new_tokens;
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
    // Устанавливаем локаль для работы с UTF-8
    std::locale::global(std::locale("en_US.UTF-8"));
    std::ios_base::sync_with_stdio(false);

    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " input_file.txt" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_filename = argv[1];
    std::ifstream infile(input_filename);
    if (!infile.is_open()){
        std::cerr << "Не удалось открыть файл: " << input_filename << std::endl;
        return EXIT_FAILURE;
    }

    // Имена выходных файлов
    std::string base_filename = input_filename.substr(0, input_filename.find_last_of('.'));
    std::string tokens_filename = base_filename + "_tokens.txt";
    std::string merges_filename = base_filename + "_merges.txt";
    
    std::ofstream tokens_file(tokens_filename);
    std::ofstream merges_file(merges_filename);
    if (!tokens_file.is_open() || !merges_file.is_open()){
        std::cerr << "Не удалось создать выходные файлы" << std::endl;
        return EXIT_FAILURE;
    }
    
    std::vector<WordEntry> words;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

    std::string line;
    int total_words = 0;
    std::cout << "Начинаем чтение файла..." << std::endl;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        std::string word_str;
        int tf, df;
        if (!(iss >> word_str >> tf >> df)) {
            std::cerr << "Ошибка чтения строки: " << line << std::endl;
            continue;
        }
        WordEntry entry;
        entry.tf = tf;
        entry.df = df;
        // Разбиваем слово на отдельные Unicode-символы
        std::wstring wide_word = converter.from_bytes(word_str);
        for (const auto &wc : wide_word) {
            std::string token = converter.to_bytes(wc);
            int token_id = get_token_id(token);
            entry.tokens.push_back(token_id);
        }
        // Вычисляем набор смежных пар для слова
        entry.pairSet = compute_pairs(entry.tokens);
        words.push_back(entry);
        total_words++;
    }
    infile.close();
    std::cout << "Прочитано " << total_words << " слов, " 
              << id_to_token.size() << " уникальных токенов (начальных)." << std::endl;

    // Глобальная мапа: пара токенов -> множество индексов слов, где пара встречается
    std::unordered_map<std::pair<int, int>, std::unordered_set<int>, IntPairHash> pair_to_word_indices;
    for (size_t i = 0; i < words.size(); i++) {
        for (const auto &p : words[i].pairSet) {
            pair_to_word_indices[p].insert(i);
        }
    }
    
    // Вектор для хранения правил слияния
    std::vector<MergeRule> merges;
    
    std::cout << "Начинаем процесс слияний..." << std::endl;
    int iteration = 0;
    while (true) {
        iteration++;
        // Поиск лучшей пары (с максимальной суммарной частотой, учитывая tf)
        std::pair<int, int> best_pair = {-1, -1};
        int best_freq = 0;
        for (const auto &kv : pair_to_word_indices) {
            const auto &p = kv.first;
            int freq = 0;
            // Для каждого слова, где пара встречается, подсчитываем количество вхождений (умножаем на tf слова)
            for (int word_idx : kv.second) {
                freq += count_occurrences_in_word(words[word_idx].tokens, p) * words[word_idx].tf;
            }
            if (freq > best_freq) {
                best_freq = freq;
                best_pair = p;
            }
        }
        if (best_freq < 2)
            break; // завершаем, если нет пар с частотой >= 2
        
        // Создаём новый токен как конкатенацию строк представлений сливаемых токенов
        std::string new_token_str = id_to_token[best_pair.first] + id_to_token[best_pair.second];
        int new_token_id = get_token_id(new_token_str);
        merges.push_back({best_pair, new_token_id, best_freq});
        std::cout << "Итерация " << iteration << ": слияние пары (" 
                  << id_to_token[best_pair.first] << ", " << id_to_token[best_pair.second] 
                  << ") -> " << new_token_str << ", частота: " << best_freq << std::endl;
        
        // Получаем все индексы слов, в которых встречается лучшая пара
        std::unordered_set<int> affected_words = pair_to_word_indices[best_pair];
        // Обновляем токенизацию только для этих слов
        for (int idx : affected_words) {
            WordEntry &word = words[idx];
            auto old_pairSet = word.pairSet; // запоминаем старый набор пар
            // Обновляем токены слова – объединяем все вхождения best_pair
            std::vector<int> new_tokens = merge_pair_in_word(word.tokens, best_pair, new_token_id);
            word.tokens = new_tokens;
            // Пересчитываем набор смежных пар для обновлённого слова
            std::unordered_set<std::pair<int, int>, IntPairHash> new_pairSet = compute_pairs(word.tokens);
            
            // Для тех пар, которые были, но исчезли – удаляем индекс слова из глобальной мапы
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
            // Для новых пар, которые появились – добавляем индекс слова
            for (const auto &p : new_pairSet) {
                if (old_pairSet.find(p) == old_pairSet.end()) {
                    pair_to_word_indices[p].insert(idx);
                }
            }
            word.pairSet = new_pairSet;
        }
        // Удаляем лучшую пару из глобальной мапы (она уже объединена во всех затронутых словах)
        pair_to_word_indices.erase(best_pair);
    }
    std::cout << "Завершено " << iteration - 1 << " слияний" << std::endl;
    
    // Подсчёт финальных частот токенов (суммируем с учётом tf каждого слова)
    std::unordered_map<int, int> final_frequencies;
    for (const auto &word : words) {
        for (int token_id : word.tokens) {
            final_frequencies[token_id] += word.tf;
        }
    }
    
    std::cout << "Найдено " << final_frequencies.size() 
              << " уникальных токенов после слияний" << std::endl;
    
    // Вывод финальных токенов с их частотами
    tokens_file << "Токен\tЧастота\n";
    for (const auto &kv : final_frequencies) {
        tokens_file << id_to_token[kv.first] << "\t" << kv.second << "\n";
    }
    
    // Вывод правил слияния
    merges_file << "Правила слияния (merges):\n";
    for (const auto &rule : merges) {
        merges_file << "(" << id_to_token[rule.merge_pair.first] << ", " 
                    << id_to_token[rule.merge_pair.second] << ") -> " 
                    << id_to_token[rule.new_token] << ", частота: " << rule.frequency << "\n";
    }
    
    tokens_file.close();
    merges_file.close();
    std::cout << "Токены записаны в файл: " << tokens_filename << std::endl;
    std::cout << "Правила слияния записаны в файл: " << merges_filename << std::endl;
    
    return EXIT_SUCCESS;
}
