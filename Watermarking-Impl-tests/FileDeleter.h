#pragma once
#include <filesystem>
#include <iostream>
#include <system_error>

/*!
 *  \brief  Helper class used in testing for deleting files automatically
 *  \author Dimitris Karatzas
 */
class FileDeleter {
    std::filesystem::path filePath;
public:
    explicit FileDeleter(const std::filesystem::path& path) : filePath(path) {}
    ~FileDeleter() 
    {
        using std::filesystem::exists;
        using std::filesystem::remove;
        std::error_code ec;
        if (exists(filePath, ec) && !remove(filePath, ec))
            std::cerr << "Warning: Failed to delete file " << filePath << ": " << ec.message() << "\n"; 
    }
    FileDeleter(const FileDeleter&) = delete;
    FileDeleter& operator=(const FileDeleter&) = delete;
};