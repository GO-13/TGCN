// ---------- Simulated Annealing ----------

namespace {
class SimulatedAnnealing : public Metaheuristic {
 public:
  SimulatedAnnealing(Solver* const s, bool maximize, IntVar* objective,
                     int64_t step, int64_t initial_temperature);
  ~SimulatedAnnealing() override {}
  void EnterSearch() override;
  void ApplyDecision(Decision* d) override;
  bool AtSolution() override;
  bool LocalOptimum() override;
  void AcceptNeighbor() override;
  std::string DebugString() const override { return "Simulated Annealing"; }

 private:
  double Temperature() const;

  const int64_t temperature0_;
  int64_t iteration_;
  std::mt19937 rand_;
  bool found_initial_solution_;

  DISALLOW_COPY_AND_ASSIGN(SimulatedAnnealing);
};

SimulatedAnnealing::SimulatedAnnealing(Solver* const s, bool maximize,
                                       IntVar* objective, int64_t step,
                                       int64_t initial_temperature)
    : Metaheuristic(s, maximize, objective, step),
      temperature0_(initial_temperature),
      iteration_(0),
      rand_(CpRandomSeed()),
      found_initial_solution_(false) {}

void SimulatedAnnealing::EnterSearch() {
  Metaheuristic::EnterSearch();
  found_initial_solution_ = false;
}

void SimulatedAnnealing::ApplyDecision(Decision* const d) {
  Solver* const s = solver();
  if (d == s->balancing_decision()) {
    return;
  }
  const double rand_double = absl::Uniform<double>(rand_, 0.0, 1.0);
#if defined(_MSC_VER) || defined(__ANDROID__)
  const double rand_log2_double = log(rand_double) / log(2.0L);
#else
  const double rand_log2_double = log2(rand_double);
#endif
  const int64_t energy_bound = Temperature() * rand_log2_double;
  if (maximize_) {
    const int64_t bound = (current_ > std::numeric_limits<int64_t>::min())
                              ? current_ + step_ + energy_bound
                              : current_;
    s->AddConstraint(s->MakeGreaterOrEqual(objective_, bound));
  } else {
    const int64_t bound = (current_ < std::numeric_limits<int64_t>::max())
                              ? current_ - step_ - energy_bound
                              : current_;
    s->AddConstraint(s->MakeLessOrEqual(objective_, bound));
  }
}

bool SimulatedAnnealing::AtSolution() {
  if (!Metaheuristic::AtSolution()) {
    return false;
  }
  found_initial_solution_ = true;
  return true;
}

bool SimulatedAnnealing::LocalOptimum() {
  if (maximize_) {
    current_ = std::numeric_limits<int64_t>::min();
  } else {
    current_ = std::numeric_limits<int64_t>::max();
  }
  ++iteration_;
  return found_initial_solution_ && Temperature() > 0;
}

void SimulatedAnnealing::AcceptNeighbor() {
  if (iteration_ > 0) {
    ++iteration_;
  }
}

double SimulatedAnnealing::Temperature() const {
  if (iteration_ > 0) {
    return (1.0 * temperature0_) / iteration_;  // Cauchy annealing
  } else {
    return 0.;
  }
}
}  // namespace

SearchMonitor* Solver::MakeSimulatedAnnealing(bool maximize, IntVar* const v,
                                              int64_t step,
                                              int64_t initial_temperature) {
  return RevAlloc(
      new SimulatedAnnealing(this, maximize, v, step, initial_temperature));
}
