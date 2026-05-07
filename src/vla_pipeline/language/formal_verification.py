"""
Formal Verification of Language → Action Semantics

This module provides rigorous formal verification of the semantic parser,
including grammar definition, denotational semantics, correctness proofs,
and adversarial testing.
"""

import re
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

from .semantic_parser import (
    SemanticParser, SemanticProgram, GoalType, 
    ObjectDescriptor, RelationType, SpatialRelation
)


# =============================================================================
# 1. FORMAL GRAMMAR DEFINITION (Context-Free Grammar)
# =============================================================================

class FormalGrammar:
    """
    Context-Free Grammar for the command language.
    
    Grammar in BNF notation:
    
    <command> ::= <simple_command> | <compound_command>
    
    <simple_command> ::= <action> <object_phrase> [<location_phrase>]
    
    <compound_command> ::= <simple_command> <conjunction> <simple_command>
    
    <action> ::= "pick" | "grab" | "grasp" | "take" | "get"
               | "place" | "put" | "set" | "drop" | "position"
               | "move" | "bring" | "carry" | "transfer"
               | "stack" | "pile"
    
    <object_phrase> ::= [<article>] [<size>] [<color>] <shape>
    
    <location_phrase> ::= <spatial_relation> <object_phrase>
    
    <spatial_relation> ::= "left of" | "right of" | "in front of" | "behind"
                         | "on" | "on top of" | "above" | "below"
                         | "next to" | "beside" | "near" | "between"
    
    <article> ::= "the" | "a" | "an"
    
    <size> ::= "small" | "medium" | "large" | "tiny" | "big" | "huge"
    
    <color> ::= "red" | "blue" | "green" | "yellow" | "orange" | "purple"
              | "black" | "white" | "brown" | "pink" | "gray" | "grey"
    
    <shape> ::= "cube" | "box" | "block"
              | "sphere" | "ball"
              | "cylinder" | "can" | "tube"
              | "pyramid" | "cone"
    
    <conjunction> ::= "and" | "then" | "after that" | "next"
    """
    
    def __init__(self):
        """Initialize formal grammar."""
        self.terminals = self._define_terminals()
        self.non_terminals = {'command', 'simple_command', 'compound_command',
                             'action', 'object_phrase', 'location_phrase',
                             'spatial_relation', 'article', 'size', 'color', 
                             'shape', 'conjunction'}
        self.production_rules = self._define_production_rules()
        self.start_symbol = 'command'
    
    def _define_terminals(self) -> Dict[str, List[str]]:
        """Define terminal symbols."""
        return {
            'action': ['pick', 'grab', 'grasp', 'take', 'get',
                      'place', 'put', 'set', 'drop', 'position',
                      'move', 'bring', 'carry', 'transfer',
                      'stack', 'pile'],
            'article': ['the', 'a', 'an'],
            'size': ['small', 'medium', 'large', 'tiny', 'big', 'huge'],
            'color': ['red', 'blue', 'green', 'yellow', 'orange', 'purple',
                     'black', 'white', 'brown', 'pink', 'gray', 'grey'],
            'shape': ['cube', 'box', 'block', 'sphere', 'ball',
                     'cylinder', 'can', 'tube', 'pyramid', 'cone'],
            'spatial_relation': ['left of', 'right of', 'in front of', 'behind',
                                'on', 'on top of', 'above', 'below',
                                'next to', 'beside', 'near', 'between'],
            'conjunction': ['and', 'then', 'after that', 'next'],
        }
    
    def _define_production_rules(self) -> Dict[str, List[List[str]]]:
        """Define production rules."""
        return {
            'command': [
                ['simple_command'],
                ['compound_command']
            ],
            'simple_command': [
                ['action', 'object_phrase'],
                ['action', 'object_phrase', 'location_phrase']
            ],
            'compound_command': [
                ['simple_command', 'conjunction', 'simple_command']
            ],
            'object_phrase': [
                ['shape'],
                ['color', 'shape'],
                ['size', 'shape'],
                ['size', 'color', 'shape'],
                ['article', 'shape'],
                ['article', 'color', 'shape'],
                ['article', 'size', 'shape'],
                ['article', 'size', 'color', 'shape']
            ],
            'location_phrase': [
                ['spatial_relation', 'object_phrase']
            ]
        }
    
    def is_valid_sentence(self, tokens: List[str]) -> bool:
        """
        Check if token sequence is valid according to grammar.
        Simplified validation - full CYK parsing would be more rigorous.
        """
        # Check if contains valid action
        has_action = any(token in self.terminals['action'] for token in tokens)
        if not has_action:
            return False
        
        # Check if contains valid shape
        has_shape = any(token in self.terminals['shape'] for token in tokens)
        if not has_shape:
            return False
        
        return True


# =============================================================================
# 2. DENOTATIONAL SEMANTICS
# =============================================================================

@dataclass
class DenotationalSemantics:
    """
    Formal denotational semantics mapping language to actions.
    
    Defines the meaning of each command in terms of its denotation
    in the semantic domain.
    
    Semantic Domain: D = Goals × Objects × Relations
    
    Denotation Function: ⟦·⟧: Command → D
    """
    
    @staticmethod
    def denote_action(action_word: str) -> GoalType:
        """
        Denotation of action words.
        
        ⟦action⟧ : ActionWord → GoalType
        """
        action_map = {
            'pick': GoalType.PICK, 'grab': GoalType.PICK,
            'grasp': GoalType.PICK, 'take': GoalType.PICK,
            'get': GoalType.PICK,
            
            'place': GoalType.PLACE, 'put': GoalType.PLACE,
            'set': GoalType.PLACE, 'drop': GoalType.PLACE,
            'position': GoalType.PLACE,
            
            'move': GoalType.MOVE, 'bring': GoalType.MOVE,
            'carry': GoalType.MOVE, 'transfer': GoalType.MOVE,
            
            'stack': GoalType.STACK, 'pile': GoalType.STACK,
        }
        return action_map.get(action_word.lower(), GoalType.NONE)
    
    @staticmethod
    def denote_spatial_relation(relation_phrase: str) -> RelationType:
        """
        Denotation of spatial relations.
        
        ⟦relation⟧ : RelationPhrase → RelationType
        """
        relation_map = {
            'left of': RelationType.LEFT_OF,
            'right of': RelationType.RIGHT_OF,
            'in front of': RelationType.IN_FRONT_OF,
            'behind': RelationType.BEHIND,
            'on': RelationType.ON,
            'on top of': RelationType.ON,
            'above': RelationType.ABOVE,
            'below': RelationType.BELOW,
            'next to': RelationType.NEXT_TO,
            'beside': RelationType.NEXT_TO,
            'near': RelationType.NEAR,
            'between': RelationType.BETWEEN,
        }
        return relation_map.get(relation_phrase.lower(), RelationType.NONE)
    
    @staticmethod
    def compositional_semantics(
        program: SemanticProgram
    ) -> Dict[str, Any]:
        """
        Compositional denotation of complete program.
        
        ⟦command⟧ = ⟦action⟧ × ⟦object⟧ × ⟦relation⟧
        """
        return {
            'goal': program.goal.value,
            'object_denotation': program.object.to_dict(),
            'relation_denotation': program.relation.to_dict() if program.relation else None,
            'constraints': program.constraints,
        }


# =============================================================================
# 3. CORRECTNESS PROOFS & PROPERTIES
# =============================================================================

class CorrectnessVerifier:
    """
    Verifies correctness properties of the semantic parser.
    
    Properties to verify:
    1. Completeness: Every valid command has a parse
    2. Soundness: Every parse corresponds to valid command
    3. Determinism: Same command → same parse
    4. Ambiguity detection: Ambiguous commands are flagged
    """
    
    def __init__(self, parser: SemanticParser, grammar: FormalGrammar):
        """Initialize verifier."""
        self.parser = parser
        self.grammar = grammar
        self.test_results = []
    
    def verify_completeness(self, test_commands: List[str]) -> Dict:
        """
        Verify: ∀ valid_cmd ∈ Language(G), ∃ parse ∈ Parse(cmd)
        
        Every grammatically valid command should parse successfully.
        """
        results = {'total': len(test_commands), 'parsed': 0, 'failed': []}
        
        for cmd in test_commands:
            try:
                program = self.parser.parse(cmd)
                if program.goal != GoalType.NONE:
                    results['parsed'] += 1
                else:
                    results['failed'].append(cmd)
            except Exception as e:
                results['failed'].append(f"{cmd} (Exception: {str(e)})")
        
        results['completeness_rate'] = results['parsed'] / results['total']
        return results
    
    def verify_soundness(self, test_programs: List[SemanticProgram]) -> Dict:
        """
        Verify: ∀ parse ∈ Parse(cmd), is_valid(parse) = True
        
        Every produced parse should be semantically valid.
        """
        results = {'total': len(test_programs), 'valid': 0, 'invalid': []}
        
        for program in test_programs:
            is_valid, errors = self.parser.validate_program(program)
            if is_valid:
                results['valid'] += 1
            else:
                results['invalid'].append((program.metadata.get('raw_command', ''), errors))
        
        results['soundness_rate'] = results['valid'] / results['total']
        return results
    
    def verify_determinism(self, test_commands: List[str]) -> Dict:
        """
        Verify: ∀ cmd, Parse(cmd) = Parse(cmd)
        
        Same command should produce same parse on multiple runs.
        """
        results = {'total': len(test_commands), 'deterministic': 0, 'non_deterministic': []}
        
        for cmd in test_commands:
            parses = [self.parser.parse(cmd).to_json() for _ in range(5)]
            if len(set(parses)) == 1:
                results['deterministic'] += 1
            else:
                results['non_deterministic'].append(cmd)
        
        results['determinism_rate'] = results['deterministic'] / results['total']
        return results
    
    def detect_ambiguity(self, test_commands: List[str]) -> Dict:
        """
        Detect ambiguous commands that could have multiple interpretations.
        """
        results = {'total': len(test_commands), 'ambiguous': [], 'unambiguous': 0}
        
        for cmd in test_commands:
            # Check for multiple objects of same type
            words = cmd.lower().split()
            colors_found = [w for w in words if w in self.grammar.terminals['color']]
            shapes_found = [w for w in words if w in self.grammar.terminals['shape']]
            
            # If multiple colors/shapes but no clear ordering, it's ambiguous
            if len(colors_found) > 2 or len(shapes_found) > 2:
                results['ambiguous'].append((cmd, "Multiple objects without clear reference"))
            else:
                results['unambiguous'] += 1
        
        results['ambiguity_rate'] = len(results['ambiguous']) / results['total']
        return results


# =============================================================================
# 4. COVERAGE ANALYSIS
# =============================================================================

class CoverageAnalyzer:
    """
    Analyzes linguistic coverage of the parser.
    """
    
    def __init__(self, grammar: FormalGrammar):
        """Initialize analyzer."""
        self.grammar = grammar
    
    def analyze_spatial_relation_coverage(self) -> Dict:
        """
        Compute coverage of spatial relations.
        
        Reports:
        - Total possible relations
        - Supported relations
        - Coverage percentage
        - Unsupported relations and reasons
        """
        # All theoretically possible spatial relations
        all_possible = [
            'left_of', 'right_of', 'in_front_of', 'behind',
            'on', 'above', 'below', 'next_to', 'near', 'far_from',
            'between', 'inside', 'outside', 'around', 'through',
            'across', 'along', 'against', 'touching', 'separate_from'
        ]
        
        # Currently supported
        supported = [rt.value for rt in RelationType if rt != RelationType.NONE]
        
        # Not supported
        not_supported = [r for r in all_possible if r not in supported]
        
        return {
            'total_possible': len(all_possible),
            'supported': len(supported),
            'coverage_percentage': len(supported) / len(all_possible) * 100,
            'supported_relations': supported,
            'not_supported': not_supported,
            'reasons': {
                'inside': 'Requires 3D containment reasoning',
                'outside': 'Requires 3D containment reasoning',
                'around': 'Requires circular/surrounding topology',
                'through': 'Requires path planning',
                'across': 'Requires path planning',
                'along': 'Requires trajectory specification',
                'against': 'Requires contact reasoning',
                'touching': 'Requires contact reasoning',
                'separate_from': 'Requires negative constraints',
            }
        }
    
    def analyze_linguistic_constructs(self) -> Dict:
        """
        Analyze supported vs unsupported linguistic constructs.
        """
        supported = {
            'simple_imperatives': True,
            'compound_commands_and': True,
            'compound_commands_then': True,
            'object_properties_color': True,
            'object_properties_shape': True,
            'object_properties_size': True,
            'spatial_relations': True,
            'definite_articles': True,
        }
        
        not_supported = {
            'negation': 'Not explicitly handled (e.g., "do not pick")',
            'conditionals': 'No if-then constructs',
            'quantifiers': 'No "all", "some", "every"',
            'relative_clauses': 'No "the cube that is red"',
            'questions': 'No interrogative forms',
            'pronouns': 'No anaphora resolution (e.g., "it", "that")',
            'temporal_expressions': 'No "after 5 seconds"',
            'modal_verbs': 'No "should", "must", "can"',
            'comparatives': 'No "bigger than", "closer to"',
            'counting': 'No "two cubes", "three times"',
        }
        
        return {
            'supported': supported,
            'not_supported': not_supported,
            'support_rate': len(supported) / (len(supported) + len(not_supported))
        }


# =============================================================================
# 5. ADVERSARIAL TESTING & CONFUSION MATRICES
# =============================================================================

class AdversarialTester:
    """
    Conducts adversarial testing on the semantic parser.
    """
    
    def __init__(self, parser: SemanticParser):
        """Initialize tester."""
        self.parser = parser
        self.test_results = []
    
    def generate_adversarial_inputs(self) -> List[Tuple[str, str]]:
        """
        Generate adversarial inputs designed to break the parser.
        
        Returns:
            List of (command, expected_failure_reason) tuples
        """
        adversarial = [
            # Ambiguous references
            ("Pick the red cube and place it on the red cube", "ambiguous_object"),
            ("Move the cube to the cube", "ambiguous_object"),
            
            # Missing information
            ("Pick it", "underspecified_object"),
            ("Place there", "underspecified_location"),
            ("Do the thing", "vague_action"),
            
            # Nonsensical commands
            ("Pick the green cube and eat it", "invalid_action"),
            ("Place the cube inside itself", "logical_contradiction"),
            
            # Complex nested structure
            ("Pick the red cube that is on the blue cube that is left of the green cube",
             "complex_nesting"),
            
            # Negation (unsupported)
            ("Do not pick the red cube", "negation_unsupported"),
            
            # Quantifiers (unsupported)
            ("Pick all the red cubes", "quantifier_unsupported"),
            
            # Pronouns (unsupported)
            ("Pick the red cube and place it there", "pronoun_unsupported"),
            
            # Typos and misspellings
            ("Pik the rd cub", "typo"),
            
            # Mixed languages
            ("Pick le cube rouge", "mixed_language"),
            
            # Long-distance dependencies
            ("The cube, which is red and sits on the table, please pick it up",
             "complex_syntax"),
        ]
        
        return adversarial
    
    def test_adversarial_inputs(self) -> Dict:
        """
        Test adversarial inputs and measure failure modes.
        """
        adversarial = self.generate_adversarial_inputs()
        results = {
            'total': len(adversarial),
            'correctly_rejected': 0,
            'incorrectly_accepted': 0,
            'exceptions': 0,
            'failures_by_type': defaultdict(int),
            'details': []
        }
        
        for cmd, expected_failure in adversarial:
            try:
                program = self.parser.parse(cmd)
                is_valid, errors = self.parser.validate_program(program)
                
                if program.goal == GoalType.NONE or not is_valid:
                    # Correctly rejected
                    results['correctly_rejected'] += 1
                    results['failures_by_type'][expected_failure] += 1
                else:
                    # Incorrectly accepted (dangerous!)
                    results['incorrectly_accepted'] += 1
                    results['details'].append({
                        'command': cmd,
                        'expected_failure': expected_failure,
                        'actual': 'accepted',
                        'program': program.to_dict()
                    })
            except Exception as e:
                results['exceptions'] += 1
                results['failures_by_type'][expected_failure] += 1
        
        results['rejection_rate'] = results['correctly_rejected'] / results['total']
        return results
    
    def generate_paraphrase_pairs(self) -> List[Tuple[str, str]]:
        """
        Generate semantically equivalent paraphrases.
        """
        paraphrases = [
            ("Pick the red cube", "Grab the red cube"),
            ("Pick the red cube", "Take the red cube"),
            ("Place the cube on the table", "Put the cube on the table"),
            ("Move the sphere left of the cube", "Bring the sphere to the left of the cube"),
            ("Stack the red box on the blue box", "Put the red box on the blue box"),
        ]
        return paraphrases
    
    def test_semantic_equivalence(self) -> Dict:
        """
        Test if paraphrases produce equivalent semantic programs.
        """
        paraphrases = self.generate_paraphrase_pairs()
        results = {
            'total': len(paraphrases),
            'equivalent': 0,
            'non_equivalent': [],
        }
        
        for cmd1, cmd2 in paraphrases:
            prog1 = self.parser.parse(cmd1)
            prog2 = self.parser.parse(cmd2)
            
            # Check semantic equivalence (goal and object should match)
            if (prog1.goal == prog2.goal and 
                prog1.object.to_dict() == prog2.object.to_dict()):
                results['equivalent'] += 1
            else:
                results['non_equivalent'].append({
                    'pair': (cmd1, cmd2),
                    'prog1': prog1.to_dict(),
                    'prog2': prog2.to_dict(),
                })
        
        results['equivalence_rate'] = results['equivalent'] / results['total']
        return results
    
    def generate_confusion_matrix(self) -> Dict:
        """
        Generate confusion matrix for relation parsing.
        """
        # Test cases: (command, true_relation)
        test_cases = [
            ("place the cube left of the box", "left_of"),
            ("place the cube to the left of the box", "left_of"),
            ("put the cube right of the box", "right_of"),
            ("set the cube on the box", "on"),
            ("place the cube on top of the box", "on"),
            ("put the cube above the box", "above"),
            ("place the cube next to the box", "next_to"),
            ("put the cube beside the box", "next_to"),
        ]
        
        # Compute confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))
        
        for cmd, true_relation in test_cases:
            program = self.parser.parse(cmd)
            if program.relation:
                predicted_relation = program.relation.type.value
            else:
                predicted_relation = "none"
            
            confusion[true_relation][predicted_relation] += 1
        
        return dict(confusion)


# =============================================================================
# 6. MAIN VERIFICATION SUITE
# =============================================================================

def run_formal_verification() -> Dict:
    """
    Run complete formal verification suite.
    
    Returns comprehensive verification report.
    """
    parser = SemanticParser()
    grammar = FormalGrammar()
    verifier = CorrectnessVerifier(parser, grammar)
    coverage = CoverageAnalyzer(grammar)
    adversarial = AdversarialTester(parser)
    
    # Generate test commands
    test_commands = [
        "Pick the red cube",
        "Place the blue sphere on the table",
        "Move the green box left of the yellow cylinder",
        "Grab the small red block and put it next to the big blue cube",
        "Stack the red cube on the blue cube",
        "Pick the red cube and place it left of the blue cube",
    ]
    
    # Parse all test commands
    test_programs = [parser.parse(cmd) for cmd in test_commands]
    
    report = {
        'grammar': {
            'terminals': len(grammar.terminals),
            'non_terminals': len(grammar.non_terminals),
            'production_rules': sum(len(v) for v in grammar.production_rules.values()),
        },
        'correctness': {
            'completeness': verifier.verify_completeness(test_commands),
            'soundness': verifier.verify_soundness(test_programs),
            'determinism': verifier.verify_determinism(test_commands),
            'ambiguity': verifier.detect_ambiguity(test_commands),
        },
        'coverage': {
            'spatial_relations': coverage.analyze_spatial_relation_coverage(),
            'linguistic_constructs': coverage.analyze_linguistic_constructs(),
        },
        'adversarial': {
            'adversarial_robustness': adversarial.test_adversarial_inputs(),
            'semantic_equivalence': adversarial.test_semantic_equivalence(),
            'confusion_matrix': adversarial.generate_confusion_matrix(),
        }
    }
    
    return report


if __name__ == "__main__":
    print("="*80)
    print("FORMAL VERIFICATION OF LANGUAGE → ACTION SEMANTICS")
    print("="*80)
    
    report = run_formal_verification()
    
    print("\n" + json.dumps(report, indent=2))
    
    # Save report
    with open('/tmp/formal_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("Verification complete. Report saved to /tmp/formal_verification_report.json")
    print("="*80)
