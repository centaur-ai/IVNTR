from . import axioms, predicates
import logging

class Task:

    def __init__(self, domain_name, task_name, requirements, types, objects,
                 predicates, functions, init, goal, actions, axioms,
                 use_metric):
        self.domain_name = domain_name
        self.task_name = task_name
        self.requirements = requirements
        self.types = types
        self.objects = objects
        self.predicates = predicates
        self.functions = functions
        self.init = init
        self.goal = goal
        self.actions = actions
        self.axioms = axioms
        self.axiom_counter = 0
        self.use_min_cost_metric = use_metric

    def add_axiom(self, parameters, condition):
        name = "new-axiom@%d" % self.axiom_counter
        self.axiom_counter += 1
        axiom = axioms.Axiom(name, parameters, len(parameters), condition)
        self.predicates.append(predicates.Predicate(name, parameters))
        self.axioms.append(axiom)
        return axiom

    def dump(self):
        logging.info("Problem %s: %s [%s]" %
              (self.domain_name, self.task_name, self.requirements))
        logging.info("Types:")
        for type in self.types:
            logging.info("  %s" % type)
        logging.info("Objects:")
        for obj in self.objects:
            logging.info("  %s" % obj)
        logging.info("Predicates:")
        for pred in self.predicates:
            logging.info("  %s" % pred)
        logging.info("Functions:")
        for func in self.functions:
            logging.info("  %s" % func)
        logging.info("Init:")
        for fact in self.init:
            logging.info("  %s" % fact)
        logging.info("Goal:")
        self.goal.dump()
        logging.info("Actions:")
        for action in self.actions:
            action.dump()
        if self.axioms:
            logging.info("Axioms:")
            for axiom in self.axioms:
                axiom.dump()


class Requirements:

    def __init__(self, requirements):
        self.requirements = requirements
        for req in requirements:
            assert req in (":strips", ":adl", ":typing", ":negation",
                           ":equality", ":negative-preconditions",
                           ":disjunctive-preconditions",
                           ":existential-preconditions",
                           ":universal-preconditions",
                           ":quantified-preconditions", ":conditional-effects",
                           ":derived-predicates", ":action-costs"), req

    def __str__(self):
        return ", ".join(self.requirements)
