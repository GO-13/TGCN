// ---------- Tabu Search ----------

class TabuSearch : public Metaheuristic {
 public:
  TabuSearch(Solver* const s, bool maximize, IntVar* objective, int64_t step,
             const std::vector<IntVar*>& vars, int64_t keep_tenure,
             int64_t forbid_tenure, double tabu_factor);
  ~TabuSearch() override {}
  void EnterSearch() override;
  void ApplyDecision(Decision* d) override;
  bool AtSolution() override;
  bool LocalOptimum() override;
  void AcceptNeighbor() override;
  std::string DebugString() const override { return "Tabu Search"; }

 protected:
  struct VarValue {
    VarValue(IntVar* const var, int64_t value, int64_t stamp)
        : var_(var), value_(value), stamp_(stamp) {}
    IntVar* const var_;
    const int64_t value_;
    const int64_t stamp_;
  };
  typedef std::list<VarValue> TabuList;

  virtual std::vector<IntVar*> CreateTabuVars();
  const TabuList& forbid_tabu_list() { return forbid_tabu_list_; }

 private:
  void AgeList(int64_t tenure, TabuList* list);
  void AgeLists();

  const std::vector<IntVar*> vars_;
  Assignment assignment_;
  int64_t last_;
  TabuList keep_tabu_list_;
  int64_t keep_tenure_;
  TabuList forbid_tabu_list_;
  int64_t forbid_tenure_;
  double tabu_factor_;
  int64_t stamp_;
  bool found_initial_solution_;

  DISALLOW_COPY_AND_ASSIGN(TabuSearch);
};

TabuSearch::TabuSearch(Solver* const s, bool maximize, IntVar* objective,
                       int64_t step, const std::vector<IntVar*>& vars,
                       int64_t keep_tenure, int64_t forbid_tenure,
                       double tabu_factor)
    : Metaheuristic(s, maximize, objective, step),
      vars_(vars),
      assignment_(s),
      last_(std::numeric_limits<int64_t>::max()),
      keep_tenure_(keep_tenure),
      forbid_tenure_(forbid_tenure),
      tabu_factor_(tabu_factor),
      stamp_(0),
      found_initial_solution_(false) {
  assignment_.Add(vars_);
}

void TabuSearch::EnterSearch() {
  Metaheuristic::EnterSearch();
  found_initial_solution_ = false;
}

void TabuSearch::ApplyDecision(Decision* const d) {
  Solver* const s = solver();
  if (d == s->balancing_decision()) {
    return;
  }
  // Aspiration criterion
  // Accept a neighbor if it improves the best solution found so far
  IntVar* aspiration = s->MakeBoolVar();
  if (maximize_) {
    s->AddConstraint(s->MakeIsGreaterOrEqualCstCt(
        objective_, CapAdd(best_, step_), aspiration));
  } else {
    s->AddConstraint(s->MakeIsLessOrEqualCstCt(objective_, CapSub(best_, step_),
                                               aspiration));
  }

  IntVar* tabu_var = nullptr;
  {
    // Creating the vector in a scope to make sure it gets deleted before
    // adding further constraints which could fail and lead to a leak.
    const std::vector<IntVar*> tabu_vars = CreateTabuVars();
    if (!tabu_vars.empty()) {
      tabu_var = s->MakeIsGreaterOrEqualCstVar(s->MakeSum(tabu_vars)->Var(),
                                               tabu_vars.size() * tabu_factor_);
    }
  }

  if (tabu_var != nullptr) {
    s->AddConstraint(
        s->MakeGreaterOrEqual(s->MakeSum(aspiration, tabu_var), int64_t{1}));
  }

  // Go downhill to the next local optimum
  if (maximize_) {
    const int64_t bound = (current_ > std::numeric_limits<int64_t>::min())
                              ? current_ + step_
                              : current_;
    s->AddConstraint(s->MakeGreaterOrEqual(objective_, bound));
  } else {
    const int64_t bound = (current_ < std::numeric_limits<int64_t>::max())
                              ? current_ - step_
                              : current_;
    s->AddConstraint(s->MakeLessOrEqual(objective_, bound));
  }

  // Avoid cost plateau's which lead to tabu cycles
  if (found_initial_solution_) {
    s->AddConstraint(s->MakeNonEquality(objective_, last_));
  }
}

std::vector<IntVar*> TabuSearch::CreateTabuVars() {
  Solver* const s = solver();

  // Tabu criterion
  // A variable in the "keep" list must keep its value, a variable in the
  // "forbid" list must not take its value in the list. The tabu criterion is
  // softened by the tabu factor which gives the number of violations to
  // the tabu criterion which is tolerated; a factor of 1 means no violations
  // allowed, a factor of 0 means all violations allowed.
  std::vector<IntVar*> tabu_vars;
  for (const VarValue& vv : keep_tabu_list_) {
    tabu_vars.push_back(s->MakeIsEqualCstVar(vv.var_, vv.value_));
  }
  for (const VarValue& vv : forbid_tabu_list_) {
    tabu_vars.push_back(s->MakeIsDifferentCstVar(vv.var_, vv.value_));
  }
  return tabu_vars;
}

bool TabuSearch::AtSolution() {
  if (!Metaheuristic::AtSolution()) {
    return false;
  }
  found_initial_solution_ = true;
  last_ = current_;

  // New solution found: add new assignments to tabu lists; this is only
  // done after the first local optimum (stamp_ != 0)
  if (0 != stamp_) {
    for (int i = 0; i < vars_.size(); ++i) {
      IntVar* const var = vars_[i];
      const int64_t old_value = assignment_.Value(var);
      const int64_t new_value = var->Value();
      if (old_value != new_value) {
        if (keep_tenure_ > 0) {
          VarValue keep_value(var, new_value, stamp_);
          keep_tabu_list_.push_front(keep_value);
        }
        if (forbid_tenure_ > 0) {
          VarValue forbid_value(var, old_value, stamp_);
          forbid_tabu_list_.push_front(forbid_value);
        }
      }
    }
  }
  assignment_.Store();

  return true;
}

bool TabuSearch::LocalOptimum() {
  AgeLists();
  if (maximize_) {
    current_ = std::numeric_limits<int64_t>::min();
  } else {
    current_ = std::numeric_limits<int64_t>::max();
  }
  return found_initial_solution_;
}

void TabuSearch::AcceptNeighbor() {
  if (0 != stamp_) {
    AgeLists();
  }
}

void TabuSearch::AgeList(int64_t tenure, TabuList* list) {
  while (!list->empty() && list->back().stamp_ < stamp_ - tenure) {
    list->pop_back();
  }
}

void TabuSearch::AgeLists() {
  AgeList(keep_tenure_, &keep_tabu_list_);
  AgeList(forbid_tenure_, &forbid_tabu_list_);
  ++stamp_;
}

class GenericTabuSearch : public TabuSearch {
 public:
  GenericTabuSearch(Solver* const s, bool maximize, IntVar* objective,
                    int64_t step, const std::vector<IntVar*>& vars,
                    int64_t forbid_tenure)
      : TabuSearch(s, maximize, objective, step, vars, 0, forbid_tenure, 1) {}
  std::string DebugString() const override { return "Generic Tabu Search"; }

 protected:
  std::vector<IntVar*> CreateTabuVars() override;
};

std::vector<IntVar*> GenericTabuSearch::CreateTabuVars() {
  Solver* const s = solver();

  // Tabu criterion
  // At least one element of the forbid_tabu_list must change value.
  std::vector<IntVar*> forbid_values;
  for (const VarValue& vv : forbid_tabu_list()) {
    forbid_values.push_back(s->MakeIsDifferentCstVar(vv.var_, vv.value_));
  }
  std::vector<IntVar*> tabu_vars;
  if (!forbid_values.empty()) {
    tabu_vars.push_back(s->MakeIsGreaterCstVar(s->MakeSum(forbid_values), 0));
  }
  return tabu_vars;
}

}  // namespace

SearchMonitor* Solver::MakeTabuSearch(bool maximize, IntVar* const v,
                                      int64_t step,
                                      const std::vector<IntVar*>& vars,
                                      int64_t keep_tenure,
                                      int64_t forbid_tenure,
                                      double tabu_factor) {
  return RevAlloc(new TabuSearch(this, maximize, v, step, vars, keep_tenure,
                                 forbid_tenure, tabu_factor));
}

SearchMonitor* Solver::MakeGenericTabuSearch(
    bool maximize, IntVar* const v, int64_t step,
    const std::vector<IntVar*>& tabu_vars, int64_t forbid_tenure) {
  return RevAlloc(
      new GenericTabuSearch(this, maximize, v, step, tabu_vars, forbid_tenure));
}
