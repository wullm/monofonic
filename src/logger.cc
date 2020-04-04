#include <logger.hh>

namespace music {

std::ofstream logger::output_file_;
log_level logger::log_level_ = log_level::off;

void logger::set_level(const log_level &level) {
  log_level_ = level;
}

log_level logger::get_level() {
  return log_level_;
}

void logger::set_output(const std::string filename) {
  if (output_file_.is_open()) {
    output_file_.close();
  }
  output_file_.open(filename, std::ofstream::out);
  assert(output_file_.is_open());
}

void logger::unset_output() {
  if (output_file_.is_open()) {
    output_file_.close();
  }
}

std::ofstream &logger::get_output() {
  return output_file_;
}

// global instantiations for different levels
logger the_logger;
log_stream flog(the_logger, log_level::fatal);
log_stream elog(the_logger, log_level::error);
log_stream wlog(the_logger, log_level::warning);
log_stream ilog(the_logger, log_level::info);
log_stream dlog(the_logger, log_level::debug);

} // namespace music
