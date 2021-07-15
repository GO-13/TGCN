// ---------- Guided Local Search ----------

typedef std::pair<int64_t, int64_t> Arc;

namespace {
// Base GLS penalties abstract class. Maintains the penalty frequency for each
// (variable, value) arc.
class GuidedLocalSearchPenalties {
 public:
  virtual ~GuidedLocalSearchPenalties() {}
  virtual bool HasValues() const = 0;
  virtual void Increment(const Arc& arc) = 0;
  virtual int64_t Value(const Arc& arc) const = 0;
  virtual void Reset() = 0;
};

// Dense GLS penalties implementation using a matrix to store penalties.
class GuidedLocalSearchPenaltiesTable : public GuidedLocalSearchPenalties {
 public:
  explicit GuidedLocalSearchPenaltiesTable(int size);
  ~GuidedLocalSearchPenaltiesTable() override {}
  bool HasValues() const override { return has_values_; }
  void Increment(const Arc& arc) override;
  int64_t Value(const Arc& arc) const override;
  void Reset() override;

 private:
  std::vector<std::vector<int64_t>> penalties_;
  bool has_values_;
};

GuidedLocalSearchPenaltiesTable::GuidedLocalSearchPenaltiesTable(int size)
    : penalties_(size), has_values_(false) {}

void GuidedLocalSearchPenaltiesTable::Increment(const Arc& arc) {
  std::vector<int64_t>& first_penalties = penalties_[arc.first];
  const int64_t second = arc.second;
  if (second >= first_penalties.size()) {
    first_penalties.resize(second + 1, 0);
  }
  ++first_penalties[second];
  has_values_ = true;
}

void GuidedLocalSearchPenaltiesTable::Reset() {
  has_values_ = false;
  for (int i = 0; i < penalties_.size(); ++i) {
    penalties_[i].clear();
  }
}

int64_t GuidedLocalSearchPenaltiesTable::Value(const Arc& arc) const {
  const std::vector<int64_t>& first_penalties = penalties_[arc.first];
  const int64_t second = arc.second;
  if (second >= first_penalties.size()) {
    return 0;
  } else {
    return first_penalties[second];
  }
}

// Sparse GLS penalties implementation using hash_map to store penalties.
class GuidedLocalSearchPenaltiesMap : public GuidedLocalSearchPenalties {
 public:
  explicit GuidedLocalSearchPenaltiesMap(int size);
  ~GuidedLocalSearchPenaltiesMap() override {}
  bool HasValues() const override { return (!penalties_.empty()); }
  void Increment(const Arc& arc) override;
  int64_t Value(const Arc& arc) const override;
  void Reset() override;

 private:
  Bitmap penalized_;
  absl::flat_hash_map<Arc, int64_t> penalties_;
};

GuidedLocalSearchPenaltiesMap::GuidedLocalSearchPenaltiesMap(int size)
    : penalized_(size, false) {}

void GuidedLocalSearchPenaltiesMap::Increment(const Arc& arc) {
  ++penalties_[arc];
  penalized_.Set(arc.first, true);
}

void GuidedLocalSearchPenaltiesMap::Reset() {
  penalties_.clear();
  penalized_.Clear();
}

int64_t GuidedLocalSearchPenaltiesMap::Value(const Arc& arc) const {
  if (penalized_.Get(arc.first)) {
    return gtl::FindWithDefault(penalties_, arc, 0);
  }
  return 0;
}

class GuidedLocalSearch : public Metaheuristic {
 public:
  GuidedLocalSearch(Solver* const s, IntVar* objective, bool maximize,
                    int64_t step, const std::vector<IntVar*>& vars,
                    double penalty_factor);
  ~GuidedLocalSearch() override {}
  bool AcceptDelta(Assignment* delta, Assignment* deltadelta) override;
  void ApplyDecision(Decision* d) override;
  bool AtSolution() override;
  void EnterSearch() override;
  bool LocalOptimum() override;
  virtual int64_t AssignmentElementPenalty(const Assignment& assignment,
                                           int index) = 0;
  virtual int64_t AssignmentPenalty(const Assignment& assignment, int index,
                                    int64_t next) = 0;
  virtual bool EvaluateElementValue(const Assignment::IntContainer& container,
                                    int64_t index, int* container_index,
                                    int64_t* penalty) = 0;
  virtual IntExpr* MakeElementPenalty(int index) = 0;
  std::string DebugString() const override { return "Guided Local Search"; }

 protected:
  struct Comparator {
    bool operator()(const std::pair<Arc, double>& i,
                    const std::pair<Arc, double>& j) {
      return i.second > j.second;
    }
  };

  int64_t Evaluate(const Assignment* delta, int64_t current_penalty,
                   const int64_t* const out_values, bool cache_delta_values);

  IntVar* penalized_objective_;
  Assignment assignment_;
  int64_t assignment_penalized_value_;
  int64_t old_penalized_value_;
  const std::vector<IntVar*> vars_;
  absl::flat_hash_map<const IntVar*, int64_t> indices_;
  const double penalty_factor_;
  std::unique_ptr<GuidedLocalSearchPenalties> penalties_;
  std::unique_ptr<int64_t[]> current_penalized_values_;
  std::unique_ptr<int64_t[]> delta_cache_;
  bool incremental_;
};

GuidedLocalSearch::GuidedLocalSearch(Solver* const s, IntVar* objective,
                                     bool maximize, int64_t step,
                                     const std::vector<IntVar*>& vars,
                                     double penalty_factor)
    : Metaheuristic(s, maximize, objective, step),
      penalized_objective_(nullptr),
      assignment_(s),
      assignment_penalized_value_(0),
      old_penalized_value_(0),
      vars_(vars),
      penalty_factor_(penalty_factor),
      incremental_(false) {
  if (!vars.empty()) {
    // TODO(user): Remove scoped_array.
    assignment_.Add(vars_);
    current_penalized_values_ = absl::make_unique<int64_t[]>(vars_.size());
    delta_cache_ = absl::make_unique<int64_t[]>(vars_.size());
    memset(current_penalized_values_.get(), 0,
           vars_.size() * sizeof(*current_penalized_values_.get()));
  }
  for (int i = 0; i < vars_.size(); ++i) {
    indices_[vars_[i]] = i;
  }
  if (absl::GetFlag(FLAGS_cp_use_sparse_gls_penalties)) {
    penalties_ = absl::make_unique<GuidedLocalSearchPenaltiesMap>(vars_.size());
  } else {
    penalties_ =
        absl::make_unique<GuidedLocalSearchPenaltiesTable>(vars_.size());
  }
}

// Add the following constraint (includes aspiration criterion):
// if minimizing,
//      objective =< Max(current penalized cost - penalized_objective - step,
//                       best solution cost - step)
// if maximizing,
//      objective >= Min(current penalized cost - penalized_objective + step,
//                       best solution cost + step)
void GuidedLocalSearch::ApplyDecision(Decision* const d) {
  if (d == solver()->balancing_decision()) {
    return;
  }
  assignment_penalized_value_ = 0;
  if (penalties_->HasValues()) {
    // Computing sum of penalties expression.
    // Scope needed to avoid potential leak of elements.
    {
      std::vector<IntVar*> elements;
      for (int i = 0; i < vars_.size(); ++i) {
        elements.push_back(MakeElementPenalty(i)->Var());
        const int64_t penalty = AssignmentElementPenalty(assignment_, i);
        current_penalized_values_[i] = penalty;
        delta_cache_[i] = penalty;
        assignment_penalized_value_ =
            CapAdd(assignment_penalized_value_, penalty);
      }
      penalized_objective_ = solver()->MakeSum(elements)->Var();
    }
    old_penalized_value_ = assignment_penalized_value_;
    incremental_ = false;
    if (maximize_) {
      IntExpr* min_pen_exp =
          solver()->MakeDifference(current_ + step_, penalized_objective_);
      IntVar* min_exp = solver()->MakeMin(min_pen_exp, best_ + step_)->Var();
      solver()->AddConstraint(
          solver()->MakeGreaterOrEqual(objective_, min_exp));
    } else {
      IntExpr* max_pen_exp =
          solver()->MakeDifference(current_ - step_, penalized_objective_);
      IntVar* max_exp = solver()->MakeMax(max_pen_exp, best_ - step_)->Var();
      solver()->AddConstraint(solver()->MakeLessOrEqual(objective_, max_exp));
    }
  } else {
    penalized_objective_ = nullptr;
    if (maximize_) {
      const int64_t bound = (current_ > std::numeric_limits<int64_t>::min())
                                ? current_ + step_
                                : current_;
      objective_->SetMin(bound);
    } else {
      const int64_t bound = (current_ < std::numeric_limits<int64_t>::max())
                                ? current_ - step_
                                : current_;
      objective_->SetMax(bound);
    }
  }
}

bool GuidedLocalSearch::AtSolution() {
  if (!Metaheuristic::AtSolution()) {
    return false;
  }
  if (penalized_objective_ != nullptr) {  // In case no move has been found
    current_ += penalized_objective_->Value();
  }
  assignment_.Store();
  return true;
}

void GuidedLocalSearch::EnterSearch() {
  Metaheuristic::EnterSearch();
  penalized_objective_ = nullptr;
  assignment_penalized_value_ = 0;
  old_penalized_value_ = 0;
  memset(current_penalized_values_.get(), 0,
         vars_.size() * sizeof(*current_penalized_values_.get()));
  penalties_->Reset();
}

// GLS filtering; compute the penalized value corresponding to the delta and
// modify objective bound accordingly.
bool GuidedLocalSearch::AcceptDelta(Assignment* delta, Assignment* deltadelta) {
  if (delta != nullptr || deltadelta != nullptr) {
    if (!penalties_->HasValues()) {
      return Metaheuristic::AcceptDelta(delta, deltadelta);
    }
    int64_t penalty = 0;
    if (!deltadelta->Empty()) {
      if (!incremental_) {
        penalty = Evaluate(delta, assignment_penalized_value_,
                           current_penalized_values_.get(), true);
      } else {
        penalty = Evaluate(deltadelta, old_penalized_value_, delta_cache_.get(),
                           true);
      }
      incremental_ = true;
    } else {
      if (incremental_) {
        for (int i = 0; i < vars_.size(); ++i) {
          delta_cache_[i] = current_penalized_values_[i];
        }
        old_penalized_value_ = assignment_penalized_value_;
      }
      incremental_ = false;
      penalty = Evaluate(delta, assignment_penalized_value_,
                         current_penalized_values_.get(), false);
    }
    old_penalized_value_ = penalty;
    if (!delta->HasObjective()) {
      delta->AddObjective(objective_);
    }
    if (delta->Objective() == objective_) {
      if (maximize_) {
        delta->SetObjectiveMin(
            std::max(std::min(CapSub(CapAdd(current_, step_), penalty),
                              CapAdd(best_, step_)),
                     delta->ObjectiveMin()));
      } else {
        delta->SetObjectiveMax(
            std::min(std::max(CapSub(CapSub(current_, step_), penalty),
                              CapSub(best_, step_)),
                     delta->ObjectiveMax()));
      }
    }
  }
  return true;
}

int64_t GuidedLocalSearch::Evaluate(const Assignment* delta,
                                    int64_t current_penalty,
                                    const int64_t* const out_values,
                                    bool cache_delta_values) {
  int64_t penalty = current_penalty;
  const Assignment::IntContainer& container = delta->IntVarContainer();
  const int size = container.Size();
  for (int i = 0; i < size; ++i) {
    const IntVarElement& new_element = container.Element(i);
    IntVar* var = new_element.Var();
    int64_t index = -1;
    if (gtl::FindCopy(indices_, var, &index)) {
      penalty = CapSub(penalty, out_values[index]);
      int64_t new_penalty = 0;
      if (EvaluateElementValue(container, index, &i, &new_penalty)) {
        penalty = CapAdd(penalty, new_penalty);
        if (cache_delta_values) {
          delta_cache_[index] = new_penalty;
        }
      }
    }
  }
  return penalty;
}

// Penalize all the most expensive arcs (var, value) according to their utility:
// utility(i, j) = cost(i, j) / (1 + penalty(i, j))
bool GuidedLocalSearch::LocalOptimum() {
  std::vector<std::pair<Arc, double>> utility(vars_.size());
  for (int i = 0; i < vars_.size(); ++i) {
    if (!assignment_.Bound(vars_[i])) {
      // Never synced with a solution, problem infeasible.
      return false;
    }
    const int64_t var_value = assignment_.Value(vars_[i]);
    const int64_t value =
        (var_value != i) ? AssignmentPenalty(assignment_, i, var_value) : 0;
    const Arc arc(i, var_value);
    const int64_t penalty = penalties_->Value(arc);
    utility[i] = std::pair<Arc, double>(arc, value / (penalty + 1.0));
  }
  Comparator comparator;
  std::sort(utility.begin(), utility.end(), comparator);
  int64_t utility_value = utility[0].second;
  penalties_->Increment(utility[0].first);
  for (int i = 1; i < utility.size() && utility_value == utility[i].second;
       ++i) {
    penalties_->Increment(utility[i].first);
  }
  if (maximize_) {
    current_ = std::numeric_limits<int64_t>::min();
  } else {
    current_ = std::numeric_limits<int64_t>::max();
  }
  return true;
}

class BinaryGuidedLocalSearch : public GuidedLocalSearch {
 public:
  BinaryGuidedLocalSearch(
      Solver* const solver, IntVar* const objective,
      std::function<int64_t(int64_t, int64_t)> objective_function,
      bool maximize, int64_t step, const std::vector<IntVar*>& vars,
      double penalty_factor);
  ~BinaryGuidedLocalSearch() override {}
  IntExpr* MakeElementPenalty(int index) override;
  int64_t AssignmentElementPenalty(const Assignment& assignment,
                                   int index) override;
  int64_t AssignmentPenalty(const Assignment& assignment, int index,
                            int64_t next) override;
  bool EvaluateElementValue(const Assignment::IntContainer& container,
                            int64_t index, int* container_index,
                            int64_t* penalty) override;

 private:
  int64_t PenalizedValue(int64_t i, int64_t j);
  std::function<int64_t(int64_t, int64_t)> objective_function_;
};

BinaryGuidedLocalSearch::BinaryGuidedLocalSearch(
    Solver* const solver, IntVar* const objective,
    std::function<int64_t(int64_t, int64_t)> objective_function, bool maximize,
    int64_t step, const std::vector<IntVar*>& vars, double penalty_factor)
    : GuidedLocalSearch(solver, objective, maximize, step, vars,
                        penalty_factor),
      objective_function_(std::move(objective_function)) {}

IntExpr* BinaryGuidedLocalSearch::MakeElementPenalty(int index) {
  return solver()->MakeElement(
      [this, index](int64_t i) { return PenalizedValue(index, i); },
      vars_[index]);
}

int64_t BinaryGuidedLocalSearch::AssignmentElementPenalty(
    const Assignment& assignment, int index) {
  return PenalizedValue(index, assignment.Value(vars_[index]));
}

int64_t BinaryGuidedLocalSearch::AssignmentPenalty(const Assignment& assignment,
                                                   int index, int64_t next) {
  return objective_function_(index, next);
}

bool BinaryGuidedLocalSearch::EvaluateElementValue(
    const Assignment::IntContainer& container, int64_t index,
    int* container_index, int64_t* penalty) {
  const IntVarElement& element = container.Element(*container_index);
  if (element.Activated()) {
    *penalty = PenalizedValue(index, element.Value());
    return true;
  }
  return false;
}

// Penalized value for (i, j) = penalty_factor_ * penalty(i, j) * cost (i, j)
int64_t BinaryGuidedLocalSearch::PenalizedValue(int64_t i, int64_t j) {
  const Arc arc(i, j);
  const int64_t penalty = penalties_->Value(arc);
  if (penalty != 0) {  // objective_function_->Run(i, j) can be costly
    const double penalized_value_fp =
        penalty_factor_ * penalty * objective_function_(i, j);
    const int64_t penalized_value =
        (penalized_value_fp <= std::numeric_limits<int64_t>::max())
            ? static_cast<int64_t>(penalized_value_fp)
            : std::numeric_limits<int64_t>::max();
    if (maximize_) {
      return -penalized_value;
    } else {
      return penalized_value;
    }
  } else {
    return 0;
  }
}

class TernaryGuidedLocalSearch : public GuidedLocalSearch {
 public:
  TernaryGuidedLocalSearch(
      Solver* const solver, IntVar* const objective,
      std::function<int64_t(int64_t, int64_t, int64_t)> objective_function,
      bool maximize, int64_t step, const std::vector<IntVar*>& vars,
      const std::vector<IntVar*>& secondary_vars, double penalty_factor);
  ~TernaryGuidedLocalSearch() override {}
  IntExpr* MakeElementPenalty(int index) override;
  int64_t AssignmentElementPenalty(const Assignment& assignment,
                                   int index) override;
  int64_t AssignmentPenalty(const Assignment& assignment, int index,
                            int64_t next) override;
  bool EvaluateElementValue(const Assignment::IntContainer& container,
                            int64_t index, int* container_index,
                            int64_t* penalty) override;

 private:
  int64_t PenalizedValue(int64_t i, int64_t j, int64_t k);
  int64_t GetAssignmentSecondaryValue(const Assignment::IntContainer& container,
                                      int index, int* container_index) const;

  const std::vector<IntVar*> secondary_vars_;
  std::function<int64_t(int64_t, int64_t, int64_t)> objective_function_;
};

TernaryGuidedLocalSearch::TernaryGuidedLocalSearch(
    Solver* const solver, IntVar* const objective,
    std::function<int64_t(int64_t, int64_t, int64_t)> objective_function,
    bool maximize, int64_t step, const std::vector<IntVar*>& vars,
    const std::vector<IntVar*>& secondary_vars, double penalty_factor)
    : GuidedLocalSearch(solver, objective, maximize, step, vars,
                        penalty_factor),
      secondary_vars_(secondary_vars),
      objective_function_(std::move(objective_function)) {
  if (!secondary_vars.empty()) {
    assignment_.Add(secondary_vars);
  }
}

IntExpr* TernaryGuidedLocalSearch::MakeElementPenalty(int index) {
  return solver()->MakeElement(
      [this, index](int64_t i, int64_t j) {
        return PenalizedValue(index, i, j);
      },
      vars_[index], secondary_vars_[index]);
}

int64_t TernaryGuidedLocalSearch::AssignmentElementPenalty(
    const Assignment& assignment, int index) {
  return PenalizedValue(index, assignment.Value(vars_[index]),
                        assignment.Value(secondary_vars_[index]));
}

int64_t TernaryGuidedLocalSearch::AssignmentPenalty(
    const Assignment& assignment, int index, int64_t next) {
  return objective_function_(index, next,
                             assignment.Value(secondary_vars_[index]));
}

bool TernaryGuidedLocalSearch::EvaluateElementValue(
    const Assignment::IntContainer& container, int64_t index,
    int* container_index, int64_t* penalty) {
  const IntVarElement& element = container.Element(*container_index);
  if (element.Activated()) {
    *penalty = PenalizedValue(
        index, element.Value(),
        GetAssignmentSecondaryValue(container, index, container_index));
    return true;
  }
  return false;
}

// Penalized value for (i, j) = penalty_factor_ * penalty(i, j) * cost (i, j)
int64_t TernaryGuidedLocalSearch::PenalizedValue(int64_t i, int64_t j,
                                                 int64_t k) {
  const Arc arc(i, j);
  const int64_t penalty = penalties_->Value(arc);
  if (penalty != 0) {  // objective_function_(i, j, k) can be costly
    const double penalized_value_fp =
        penalty_factor_ * penalty * objective_function_(i, j, k);
    const int64_t penalized_value =
        (penalized_value_fp <= std::numeric_limits<int64_t>::max())
            ? static_cast<int64_t>(penalized_value_fp)
            : std::numeric_limits<int64_t>::max();
    if (maximize_) {
      return -penalized_value;
    } else {
      return penalized_value;
    }
  } else {
    return 0;
  }
}

int64_t TernaryGuidedLocalSearch::GetAssignmentSecondaryValue(
    const Assignment::IntContainer& container, int index,
    int* container_index) const {
  const IntVar* secondary_var = secondary_vars_[index];
  int hint_index = *container_index + 1;
  if (hint_index > 0 && hint_index < container.Size() &&
      secondary_var == container.Element(hint_index).Var()) {
    *container_index = hint_index;
    return container.Element(hint_index).Value();
  } else {
    return container.Element(secondary_var).Value();
  }
}
}  // namespace

SearchMonitor* Solver::MakeGuidedLocalSearch(
    bool maximize, IntVar* const objective,
    Solver::IndexEvaluator2 objective_function, int64_t step,
    const std::vector<IntVar*>& vars, double penalty_factor) {
  return RevAlloc(new BinaryGuidedLocalSearch(
      this, objective, std::move(objective_function), maximize, step, vars,
      penalty_factor));
}

SearchMonitor* Solver::MakeGuidedLocalSearch(
    bool maximize, IntVar* const objective,
    Solver::IndexEvaluator3 objective_function, int64_t step,
    const std::vector<IntVar*>& vars,
    const std::vector<IntVar*>& secondary_vars, double penalty_factor) {
  return RevAlloc(new TernaryGuidedLocalSearch(
      this, objective, std::move(objective_function), maximize, step, vars,
      secondary_vars, penalty_factor));
}
