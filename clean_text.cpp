// g++ -std=c++11 -pthread -o process_files process_files.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <codecvt>
#include <locale>
#include <functional>
#include <algorithm>
#include <cwctype>

// Проверяет, является ли широкий символ латинской буквой.
// Покрываем Basic Latin, Latin-1 Supplement, Latin Extended-A/B и Latin Extended Additional.
bool isLatinChar(wchar_t ch) {
    if ((ch >= L'A' && ch <= L'Z') || (ch >= L'a' && ch <= L'z'))
        return true;
    if ((ch >= 0x00C0 && ch <= 0x00D6) ||
        (ch >= 0x00D8 && ch <= 0x00F6) ||
        (ch >= 0x00F8 && ch <= 0x00FF))
        return true;
    if (ch >= 0x0100 && ch <= 0x017F)
        return true;
    if (ch >= 0x0180 && ch <= 0x024F)
        return true;
    if (ch >= 0x1E00 && ch <= 0x1EFF)
        return true;
    return false;
}

// Проверяет, является ли широкий символ кириллической буквой.
// Покрываем Basic Cyrillic (U+0400–U+04FF) и Cyrillic Supplement (U+0500–U+052F).
bool isCyrillicChar(wchar_t ch) {
    if ((ch >= 0x0400 && ch <= 0x04FF) ||
        (ch >= 0x0500 && ch <= 0x052F))
        return true;
    return false;
}

// Универсальная функция: конвертирует UTF-8 слово в широкую строку и возвращает true,
// если каждый символ удовлетворяет предикату isScriptChar.
bool isScriptWord(const std::string &word, const std::function<bool(wchar_t)> &isScriptChar) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wword;
    try {
        wword = converter.from_bytes(word);
    } catch (const std::range_error &e) {
        // Ошибка конвертации: считаем слово недопустимым.
        return false;
    }
    for (wchar_t wc : wword) {
        if (!isScriptChar(wc))
            return false;
    }
    return true;
}

bool isLatinWord(const std::string &word) {
    return isScriptWord(word, isLatinChar);
}

bool isCyrillicWord(const std::string &word) {
    return isScriptWord(word, isCyrillicChar);
}

// Функция разбивает строку на слова, используя преобразование в широкий формат.
// Последовательность символов считается словом, если для символа выполняется условие iswalnum или символ '_'.
std::vector<std::string> filterWords(const std::string &text,
                                       const std::function<bool(const std::string&)> &wordFilter) {
    std::vector<std::string> result;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    std::wstring wtext;
    try {
        wtext = converter.from_bytes(text);
    } catch (...) {
        return result;
    }
    
    std::wstring current;
    for (wchar_t wc : wtext) {
        // Если символ является буквой/цифрой или это символ '_'
        if (std::iswalnum(wc) || wc == L'_') {
            current.push_back(wc);
        } else {
            if (!current.empty()) {
                std::string word = converter.to_bytes(current);
                if (wordFilter(word))
                    result.push_back(word);
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        std::string word = converter.to_bytes(current);
        if (wordFilter(word))
            result.push_back(word);
    }
    return result;
}

int main(int argc, char* argv[]) {
    // Устанавливаем системную локаль (важно для корректной работы iswalnum и конвертации)
    std::locale::global(std::locale(""));
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <latin|cyrillic>\n";
        return 1;
    }
    
    std::string input_filename = argv[1];
    std::string mode = argv[2];
    
    // Выбираем функцию фильтрации слов в зависимости от режима.
    std::function<bool(const std::string&)> wordFilter;
    if (mode == "latin") {
        wordFilter = isLatinWord;
    } else if (mode == "cyrillic") {
        wordFilter = isCyrillicWord;
    } else {
        std::cerr << "Invalid mode. Use 'latin' or 'cyrillic'.\n";
        return 1;
    }
    
    // Открываем и считываем все строки из входного файла.
    std::ifstream infile(input_filename, std::ios::binary | std::ios::ate);
    if (!infile) {
        std::cerr << "Failed to open input file: " << input_filename << "\n";
        return 1;
    }
    auto file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<std::string> lines;
    lines.reserve(file_size / 80); // примерная оценка для уменьшения перевыделения памяти

    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }
    infile.close();
    
    // Определяем число потоков.
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::cout << "Using " << num_threads << " threads for processing.\n";
    
    // Каждый поток будет записывать результаты в свой вектор.
    std::vector<std::vector<std::string>> thread_outputs(num_threads);
    
    // Лямбда-функция-воркер, обрабатывающая диапазон строк.
    auto worker = [&](size_t start, size_t end, unsigned int thread_index) {
        for (size_t i = start; i < end; ++i) {
            const std::string &curr_line = lines[i];
            // Извлекаем слова из строки с учетом выбранного фильтра.
            auto filtered_words = filterWords(curr_line, wordFilter);
            // Пропускаем строку, если после фильтрации осталось менее 10 слов.
            if (filtered_words.size() < 10)
                continue;
            // Собираем строку обратно (слова разделяем пробелами).
            std::ostringstream oss;
            for (size_t j = 0; j < filtered_words.size(); ++j) {
                if (j > 0) oss << " ";
                oss << filtered_words[j];
            }
            thread_outputs[thread_index].push_back(oss.str());
        }
    };
    
    // Разбиваем строки на чанки и запускаем потоки.
    size_t total_lines = lines.size();
    size_t chunk_size = (total_lines + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    
    for (unsigned int t = 0; t < num_threads; t++) {
        size_t start = t * chunk_size;
        size_t end = std::min(total_lines, start + chunk_size);
        threads.emplace_back(worker, start, end, t);
    }
    
    // Ждем завершения всех потоков.
    for (auto &t : threads) {
        if (t.joinable())
            t.join();
    }
    
    // Формируем имя выходного файла: заменяем расширение на ".step1" или добавляем его,
    // и добавляем режим обработки.
    std::string output_filename = input_filename;
    size_t pos = output_filename.rfind('.');
    if (pos != std::string::npos) {
        output_filename = output_filename.substr(0, pos) + "." + mode + ".step1";
    } else {
        output_filename += "." + mode + ".step1";
    }
    
    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << output_filename << "\n";
        return 1;
    }
    
    // Записываем отфильтрованные строки в выходной файл.
    for (unsigned int t = 0; t < num_threads; t++) {
        for (const auto &filtered_line : thread_outputs[t]) {
            outfile << filtered_line << "\n";
        }
    }
    outfile.close();
    
    std::cout << "Processing complete. Output written to " << output_filename << "\n";
    return 0;
}
