#include <logger.hh>

namespace music {

std::ofstream Logger::output_file_;
LogLevel Logger::log_level_ = LogLevel::Off;

void Logger::SetLevel(const LogLevel &level) {
  log_level_ = level;
}

LogLevel Logger::GetLevel() {
  return log_level_;
}

void Logger::SetOutput(const std::string filename) {
  if (output_file_.is_open()) {
    output_file_.close();
  }
  output_file_.open(filename, std::ofstream::out);
  assert(output_file_.is_open());
}

void Logger::UnsetOutput() {
  if (output_file_.is_open()) {
    output_file_.close();
  }
}

std::ofstream &Logger::GetOutput() {
  return output_file_;
}

// global instantiations for different levels
Logger glogger;
LogStream flog(glogger, LogLevel::Fatal);
LogStream elog(glogger, LogLevel::Error);
LogStream wlog(glogger, LogLevel::Warning);
LogStream ilog(glogger, LogLevel::Info);
LogStream dlog(glogger, LogLevel::Debug);

} // namespace music
