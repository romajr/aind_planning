"""
####### AIND Artificial Intelligence Nanodegree - Udacity #######
#---------------------------- Roma -----------------------------#
#
# This code implements a planning engine for the following
# problem:
# You have a logistic situation in a central control of airports
# and you have to plan the load/unload actions of cargos
# into airplanes from different airports to others
#
#---------------------------------------------------------------#
#################################################################
"""
from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache

class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """
        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # Function definition for the Loading action
        # The code gets all airports availables, all planes, all cargos
        # and set some predefinitions as followed:
        # 1. the cargo must be on departing airport (from)
        # 2. the plane must be on departing airport (from)
        # 3. the effect will be ADD the cargo into the plane
        # 4. the effect will be to REMOVE the cargo from the airport
        def load_actions():
            """Create all concrete Load actions and return a list
            :return: list of Action objects
            """
            loads = []
            for fr in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        # 1. condition & 2. condition
                        precond_pos = [expr("At({}, {})".format(c, fr)), expr("At({}, {})".format(p, fr))]
                        # No negative condition
                        precond_neg = []
                        # 3. condition
                        effect_add = [expr("In({}, {})".format(c, p))]
                        # 4. condition
                        effect_rem = [expr("At({}, {})".format(c, fr))]

                        # Defining the action of loading the Cargo to the Plane at the Airport
                        load = Action(expr("Load({}, {}, {})".format(c, p, fr)),
                                [precond_pos, precond_neg],
                                [effect_add, effect_rem])
                        loads.append(load)
            return loads

        # Function definition for the UnLoading action
        # The code gets all airports availables, all planes, all cargos
        # and set some predefinitions as followed:
        # 1. the cargo must be inside the plane
        # 2. the plane must be at the ARRIVING airport (to)
        # 3. the effect will be to ADD the cargo from the plane to the airport
        # 4. the effect will be to REMOVE the cargo from the plane
        def unload_actions():
            """Create all concrete Unload actions and return a list
            :return: list of Action objects
            """
            unloads = []
            for to in self.airports:
                for p in self.planes:
                    for c in self.cargos:
                        # 1. condition & 2. condition
                        precond_pos = [expr("In({}, {})".format(c, p)), expr("At({}, {})".format(p, to))]
                        # No negative condition
                        precond_neg = []
                        # 3. condition
                        effect_add = [expr("At({}, {})".format(c, to))]
                        # 4. condition
                        effect_rem = [expr("In({}, {})".format(c, p))]

                        # Defining the action of unloading the Cargo from the Plane at the Airport
                        load = Action(expr("Unload({}, {}, {})".format(c, p, to)),
                                [precond_pos, precond_neg],
                                [effect_add, effect_rem])
                        unloads.append(load)
            return unloads

        # Function definition for the Flying action
        # The code gets all airports availables (departure/arriving)
        # eliminates the cases where airports are equal and define action
        # 1. the plane must be at the DEPARTURE
        # 2. the effect will be to ADD the plane to the arriving airport (to)
        # 3. the effect will be to REMOVE the plane from the departure airport (from)
        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            # 1. condition
                            precond_pos = [expr("At({}, {})".format(p, fr))]
                            # No negative condition
                            precond_neg = []
                            # 2. condition
                            effect_add = [expr("At({}, {})".format(p, to))]
                            # 3. condition
                            effect_rem = [expr("At({}, {})".format(p, fr))]

                            # Defining the action of flying the Plane from one airport to another
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys
        return load_actions() + unload_actions() + fly_actions()

    # Function definition for Actions
    # It verifies all actions availables in our list
    # and all preconditions we set for that action (Load, Unload or Flying)
    # If the action is possible, we add it to the list
    # and return it as a list of possible actions
    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.
        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        possible_actions = []
        fluent_state = decode_state(state, self.state_map)
        for action in self.actions_list:
            if all(precond_pos in fluent_state.pos for precond_pos in action.precond_pos) and \
                    all(precond_neg in fluent_state.neg for precond_neg in action.precond_neg):
                possible_actions.append(action)
        return possible_actions

    # Function definition for Result
    # It creates the following state encoding it
    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).
        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        fluent_state = decode_state(state, self.state_map)
        pos = [fluent for fluent in fluent_state.pos
               if fluent not in action.effect_rem] + action.effect_add
        neg = [fluent for fluent in fluent_state.neg
               if fluent not in action.effect_add] + action.effect_rem
        new_state = FluentState(pos, neg)
        return encode_state(new_state, self.state_map)

    # Function definition for Goal testing
    # It decodes the state and verify the clauses to achieve the goal
    # If there is no clauses on goals list, it returns False
    # If goals achieved, returns True
    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached
        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # Requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # Function definition for Actions
        count = len(self.goal)
        state_decode = decode_state(node.state, self.state_map)
        for precond in state_decode.pos:
            if precond in self.goal:
                count -= 1
        return count

# Defining condition for Air Cargo Problem #nb.1
def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

# Defining condition for Air Cargo Problem #nb.2
def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['ATL', 'JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
            expr('At(C2, JFK)'),
            expr('At(C3, ATL)'),
            expr('At(P1, SFO)'),
            expr('At(P2, JFK)'),
            expr('At(P3, ATL)'),
           ]
    neg = [expr('At(C3, JFK)'),
            expr('At(C3, SFO)'),
            expr('In(C3, P1)'),
            expr('In(C3, P2)'),
            expr('In(C3, P3)'),
            expr('At(C2, SFO)'),
            expr('At(C2, ATL)'),
            expr('In(C2, P1)'),
            expr('In(C2, P2)'),
            expr('In(C2, P3)'),
            expr('At(C1, JFK)'),
            expr('At(C1, ATL)'),
            expr('In(C1, P1)'),
            expr('In(C1, P2)'),
            expr('In(C1, P3)'),
            expr('At(P1, JFK)'),
            expr('At(P1, ATL)'),
            expr('At(P2, SFO)'),
            expr('At(P2, ATL)'),
            expr('At(P3, JFK)'),
            expr('At(P3, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)

# Defining condition for Air Cargo Problem #nb.3
def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['ATL', 'JFK', 'SFO', 'ORD']
    pos = [expr('At(C1, SFO)'),
            expr('At(C2, JFK)'),
            expr('At(C3, ATL)'),
            expr('At(C4, ORD)'),
            expr('At(P1, SFO)'),
            expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C4, ATL)'),
            expr('At(C4, JFK)'),
            expr('At(C4, SFO)'),
            expr('In(C4, P1)'),
            expr('In(C4, P2)'),
            expr('At(C3, JFK)'),
            expr('At(C3, SFO)'),
            expr('At(C3, ORD)'),
            expr('In(C3, P1)'),
            expr('In(C3, P2)'),
            expr('At(C2, SFO)'),
            expr('At(C2, ATL)'),
            expr('At(C2, ORD)'),
            expr('In(C2, P1)'),
            expr('In(C2, P2)'),
            expr('At(C1, JFK)'),
            expr('At(C1, ATL)'),
            expr('At(C1, ORD)'),
            expr('In(C1, P1)'),
            expr('In(C1, P2)'),
            expr('At(P1, JFK)'),
            expr('At(P1, ATL)'),
            expr('At(P1, ORD)'),
            expr('At(P2, SFO)'),
            expr('At(P2, ATL)'),
            expr('At(P2, ORD)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
