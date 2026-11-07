"""DBoW2 hierarchical ORB quantization and ORB-SLAM3 inverted retrieval.

Derived from Thirdparty/DBoW2 and src/KeyFrameDatabase.cc. DBoW2: copyright
Dorian Gálvez-López (BSD); ORB-SLAM3: University of Zaragoza contributors,
GPL-3.0-or-later. See THIRD_PARTY_NOTICES.md for retained notices.

This loads the actual trained vocabulary. It does not train a replacement
codebook or substitute descriptor hashing for place recognition.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import math
import numpy as np
from .features import hamming


class Vocabulary:
    @classmethod
    def from_text(cls, path):
        """Load the original DBoW2 text format, preserving node/word order."""
        path = Path(path)
        # Two streaming passes avoid a million Python node/descriptor objects.
        with path.open() as stream:
            header = stream.readline().split()
            if len(header) != 4:
                raise ValueError("DBoW2 header must contain k, L, scoring, weighting")
            branching, depth, scoring, weighting = map(int, header)
            if not (1 <= branching <= 20 and 1 <= depth <= 10 and
                    0 <= scoring <= 5 and 0 <= weighting <= 3):
                raise ValueError("invalid DBoW2 header")
            count = 1 + sum(bool(line.strip()) for line in stream)
        result = cls()
        result.branching, result.depth = branching, depth
        result.scoring, result.weighting = scoring, weighting
        result.descriptors = np.zeros((count,32), dtype=np.uint8)
        result.children = np.zeros((count,branching), dtype=np.int32)
        result.child_counts = np.zeros(count, dtype=np.uint8)
        result.parents = np.zeros(count, dtype=np.int32)
        result.word_ids = np.full(count,-1, dtype=np.int32)
        result.weights = np.zeros(count, dtype=np.float64)
        leaves = np.zeros(count, dtype=bool)
        word_count, node = 0, 1
        with path.open() as stream:
            next(stream)
            for line_number, line in enumerate(stream, start=2):
                if not line.strip():
                    continue
                values = np.fromstring(line, sep=" ")
                if len(values) != 35 or not np.isfinite(values).all():
                    raise ValueError(f"invalid vocabulary node at line {line_number}")
                parent, leaf = int(values[0]), int(values[1])
                descriptor = values[2:34]
                if not (0 <= parent < node and leaf in (0,1)):
                    raise ValueError(f"invalid tree parent/leaf at line {line_number}")
                if np.any((descriptor < 0) | (descriptor > 255) | (descriptor != np.floor(descriptor))):
                    raise ValueError(f"invalid ORB byte at line {line_number}")
                child_count = int(result.child_counts[parent])
                if child_count == branching:
                    raise ValueError("vocabulary node exceeds declared branching factor")
                result.children[parent,child_count] = node
                result.child_counts[parent] += 1
                result.parents[node] = parent
                result.descriptors[node] = descriptor
                result.weights[node] = values[-1]
                leaves[node] = bool(leaf)
                if leaf:
                    result.word_ids[node] = word_count
                    word_count += 1
                node += 1
        if not word_count or np.any(leaves & (result.child_counts > 0)):
            raise ValueError("vocabulary has no words or a leaf has children")
        if np.any(~leaves & (result.child_counts == 0)):
            raise ValueError("vocabulary contains an empty internal node")
        result.word_count = word_count
        return result

    def __len__(self):
        return self.word_count

    def quantize(self, descriptor, levelsup=4):
        """Follow nearest Hamming child at each level; first child wins ties."""
        descriptor = np.asarray(descriptor, dtype=np.uint8)
        if descriptor.shape != (32,):
            raise ValueError("an ORB descriptor must have shape (32,)")
        node, current_depth, feature_node = 0, 0, 0
        feature_level = self.depth-int(levelsup)
        while self.child_counts[node]:
            children = self.children[node,:self.child_counts[node]]
            node = int(children[np.argmin(hamming(self.descriptors[children],descriptor))])
            current_depth += 1
            if current_depth == feature_level:
                feature_node = node
        return int(self.word_ids[node]), float(self.weights[node]), feature_node

    def transform(self, descriptors, levelsup=4):
        """Return normalized BowVector and ancestor-node FeatureVector.

        Weighting codes are DBoW2's TF_IDF=0, TF=1, IDF=2, BINARY=3.
        The file already stores trained leaf weights, including unit weights
        for TF/BINARY vocabularies. A zero leaf weight stops that word.
        """
        descriptors = np.asarray(descriptors)
        if descriptors.ndim != 2 or descriptors.shape[1] != 32 or descriptors.dtype != np.uint8:
            raise ValueError("descriptors must be uint8 with shape (N,32)")
        bow, feature_nodes = {}, defaultdict(list)
        for index, descriptor in enumerate(descriptors):
            word, weight, node = self.quantize(descriptor, levelsup)
            if weight <= 0:
                continue
            if self.weighting in (0,1):
                bow[word] = bow.get(word,0.0)+weight
            else:
                bow.setdefault(word,weight)
            feature_nodes[node].append(index)
        if bow:
            if self.scoring == 5:  # dot-product does not request normalization
                divisor = len(bow) if self.weighting in (0,1) else 1
            elif self.scoring == 1:
                divisor = math.sqrt(sum(v*v for v in bow.values()))
            else:
                divisor = sum(abs(v) for v in bow.values())
            if divisor > 0:
                bow = {word:weight/divisor for word,weight in sorted(bow.items())}
        return bow, dict(sorted(feature_nodes.items()))

    def score(self, a, b):
        shared = a.keys() & b.keys()
        if self.scoring == 0:
            return -0.5*sum(abs(a[k]-b[k])-abs(a[k])-abs(b[k]) for k in shared)
        if self.scoring == 1:
            product = sum(a[k]*b[k] for k in shared)
            return 1.0 if product >= 1 else 1-math.sqrt(1-product)
        if self.scoring == 2:
            return 2*sum(a[k]*b[k]/(a[k]+b[k]) for k in shared if a[k]+b[k] != 0)
        if self.scoring == 3:
            log_eps = math.log(np.finfo(float).eps)
            return sum(v*(math.log(v)-math.log(b[k]) if k in b and b[k] else math.log(v)-log_eps)
                       for k,v in a.items() if v)
        if self.scoring == 4:
            return sum(math.sqrt(a[k]*b[k]) for k in shared)
        return sum(a[k]*b[k] for k in shared)


@dataclass
class DatabaseEntry:
    bow: dict
    map_id: int
    # Ordered by decreasing covisibility weight, as GetBestCovisibilityKeyFrames.
    neighbors: list = field(default_factory=list)
    bad: bool = False
    place_score: float = 0.0
    reloc_score: float = 0.0


class InvertedDatabase:
    """Keyframe IDs index actual BoW words; the map owns geometry/lifetimes."""
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.entries = {}
        self.inverted = defaultdict(dict)  # ordered sets preserve insertion order
        self.bad_maps = set()

    def add(self, keyframe_id, bow, map_id, neighbors=()):
        if keyframe_id in self.entries:
            raise ValueError("keyframe already indexed")
        self.entries[keyframe_id] = DatabaseEntry(dict(bow), map_id, list(neighbors))
        for word in sorted(bow):
            self.inverted[word][keyframe_id] = None

    def erase(self, keyframe_id):
        entry = self.entries.pop(keyframe_id)
        for word in entry.bow:
            self.inverted[word].pop(keyframe_id, None)
            if not self.inverted[word]:
                del self.inverted[word]

    def clear_map(self, map_id):
        for keyframe_id in list(self.entries):
            if self.entries[keyframe_id].map_id == map_id:
                self.erase(keyframe_id)

    def _scores(self, bow, excluded, kind):
        counts = Counter()
        for word in sorted(bow):
            for keyframe_id in self.inverted.get(word,{}):
                if keyframe_id not in excluded:
                    counts[keyframe_id] += 1
        if not counts:
            return []
        minimum_words = int(max(counts.values())*np.float32(0.8))
        scores = {}
        score_attribute = "place_score" if kind == "place" else "reloc_score"
        for keyframe_id, count in counts.items():
            if count > minimum_words:
                score = self.vocabulary.score(bow,self.entries[keyframe_id].bow)
                scores[keyframe_id] = score
                setattr(self.entries[keyframe_id],score_attribute,score)
        accumulated = []
        for keyframe_id, score in scores.items():
            best_id, best_score, total = keyframe_id, score, score
            for neighbor in self.entries[keyframe_id].neighbors[:10]:
                if neighbor in counts and neighbor in self.entries:
                    # The source retains old score values for shared-word
                    # neighbors below the current 0.8 max-word threshold.
                    neighbor_score = getattr(self.entries[neighbor],score_attribute)
                    total += neighbor_score
                    if neighbor_score > best_score:
                        best_id, best_score = neighbor, neighbor_score
            accumulated.append((total,best_id))
        return accumulated

    def detect_n_best(self, bow, map_id, connected=(), count=3):
        """DetectNBestCandidates: independently cap loop and merge candidates."""
        accumulated = sorted(self._scores(bow,set(connected),"place"),
                             key=lambda pair: pair[0], reverse=True)
        loops, merges, seen = [], [], set()
        for _, keyframe_id in accumulated:
            entry = self.entries[keyframe_id]
            if entry.bad or keyframe_id in seen:
                continue
            if entry.map_id == map_id and len(loops) < count:
                loops.append(keyframe_id)
            elif entry.map_id != map_id and len(merges) < count and entry.map_id not in self.bad_maps:
                merges.append(keyframe_id)
            seen.add(keyframe_id)
            if len(loops) >= count and len(merges) >= count:
                break
        return loops, merges

    def relocalization_candidates(self, bow, map_id):
        accumulated = self._scores(bow,set(),"reloc")
        if not accumulated:
            return []
        threshold = np.float32(0.75)*max(score for score,_ in accumulated)
        retained = []
        for score,keyframe_id in accumulated:
            if (score > threshold and self.entries[keyframe_id].map_id == map_id
                    and keyframe_id not in retained):
                retained.append(keyframe_id)
        return retained
