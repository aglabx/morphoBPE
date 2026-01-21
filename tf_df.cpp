// g++ -std=c++11 -pthread -o tfdf tfdf.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <tuple>
#include <functional>
#include <mutex>

using namespace std;

// Определяем тип для хранения tf и df: pair.first = tf, pair.second = df
using FrequencyMap = unordered_map<string, pair<size_t, size_t>>;

// Функция для разбиения строки по пробельным символам (так как файл уже чистый)
vector<string> split_line(const string &line) {
    vector<string> words;
    istringstream iss(line);
    string word;
    while (iss >> word) {
        words.push_back(word);
    }
    return words;
}

// Функция-воркер для обработки диапазона строк [start, end)
// Для каждой строки считаем tf и df и записываем в локальную карту local_map.
void process_lines(const vector<string>& lines, size_t start, size_t end, FrequencyMap &local_map) {
    for (size_t i = start; i < end; i++) {
        const string &line = lines[i];
        // Разбиваем строку на слова (уже очищенные)
        vector<string> words = split_line(line);
        // Для подсчёта df используем множество уникальных слов этой строки.
        unordered_set<string> unique_words;
        for (auto &word : words) {
            // Преобразуем слово в нижний регистр
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            // Увеличиваем общее число вхождений (tf)
            local_map[word].first++;
            unique_words.insert(word);
        }
        // Для каждого уникального слова увеличиваем счетчик df (число строк, где оно встречалось)
        for (const auto &word : unique_words) {
            local_map[word].second++;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <clean_file>\n";
        return 1;
    }
    
    string input_filename = argv[1];
    
    // Читаем все строки из входного файла
    ifstream infile(input_filename);
    if (!infile) {
        cerr << "Failed to open input file: " << input_filename << "\n";
        return 1;
    }
    
    vector<string> lines;
    string line;
    while (getline(infile, line)) {
        if (!line.empty()) { // пропускаем пустые строки, если есть
            lines.push_back(line);
        }
    }
    infile.close();
    
    // Определяем количество потоков (если hardware_concurrency возвращает 0, то ставим 4)
    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    cout << "Using " << num_threads << " threads.\n";
    
    // Создаём вектор локальных карт для каждого потока
    vector<FrequencyMap> local_maps(num_threads);
    vector<thread> threads;
    size_t total_lines = lines.size();
    size_t chunk_size = (total_lines + num_threads - 1) / num_threads;
    
    // Запускаем потоки, каждому назначаем свой диапазон строк
    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = min(total_lines, start + chunk_size);
        threads.emplace_back(process_lines, cref(lines), start, end, ref(local_maps[t]));
    }
    
    // Ждем завершения всех потоков
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }
    
    // Объединяем локальные карты в одну глобальную
    FrequencyMap global_map;
    for (unsigned int t = 0; t < num_threads; t++) {
        for (const auto &entry : local_maps[t]) {
            global_map[entry.first].first  += entry.second.first;  // tf
            global_map[entry.first].second += entry.second.second; // df
        }
    }
    
    // Формируем вектор для сортировки: каждый элемент — кортеж (слово, tf, df)
    vector<tuple<string, size_t, size_t>> word_stats;
    word_stats.reserve(global_map.size());
    for (const auto &entry : global_map) {
        word_stats.emplace_back(entry.first, entry.second.first, entry.second.second);
    }
    
    // Сортируем по tf по убыванию; при равном tf — по слову в лексикографическом порядке
    sort(word_stats.begin(), word_stats.end(), 
        [](const tuple<string, size_t, size_t> &a, 
           const tuple<string, size_t, size_t> &b) {
        if (get<1>(a) != get<1>(b))
            return get<1>(a) > get<1>(b);
        return get<0>(a) < get<0>(b);
    });
    
    // Формируем имя выходного файла, например, добавляя расширение ".tfdf.tsv"
    string output_filename = input_filename + ".tfdf.tsv";
    ofstream outfile(output_filename);
    if (!outfile) {
        cerr << "Failed to open output file: " << output_filename << "\n";
        return 1;
    }
    
    // Записываем результат в файл: слово, tf, df (табуляция между полями)
    for (const auto &entry : word_stats) {
        outfile << get<0>(entry) << "\t" << get<1>(entry) << "\t" << get<2>(entry) << "\n";
    }
    outfile.close();
    
    cout << "Processing complete. Output written to " << output_filename << "\n";
    return 0;
}
