from rapidfuzz import fuzz


class NeedlemanAligner:
    """
    Production Anchored Needleman-Wunsch Alignment

    Pipeline

        TimestampMatcher
                │
                ▼
          Anchor Extraction
                │
                ▼
         Window Segmentation
                │
                ▼
       Local Needleman Alignment
                │
                ▼
         Merge Window Results

    This implementation never performs a global
    Needleman over the entire transcript.

    Instead, timestamp matches become fixed anchors
    while Needleman is only responsible for aligning
    uncertain regions between anchors.

    Advantages

    ✔ Faster
    ✔ Better medical term preservation
    ✔ Less drift
    ✔ Scales to long conversations
    """

    def __init__(
        self,
        match_score=2,
        similar_score=1,
        mismatch_score=-1,
        gap_score=-2,
        timestamp_bonus=0.5,
        confidence_bonus=0.5,
        similarity_threshold=90,
    ):

        self.match_score = match_score
        self.similar_score = similar_score
        self.mismatch_score = mismatch_score
        self.gap_score = gap_score

        self.timestamp_bonus = timestamp_bonus
        self.confidence_bonus = confidence_bonus

        self.similarity_threshold = similarity_threshold
        
    def score(self, whisper, medasr):

        if whisper is None or medasr is None:
            return self.gap_score

        score = self.mismatch_score

        if whisper["token"] == medasr["token"]:
            score = self.match_score

        else:
            sim = fuzz.ratio(
                whisper["token"],
                medasr["token"]
            )

            if sim >= self.similarity_threshold:
                score = self.similar_score

        if abs(
            whisper["start"] -
            medasr["start"]
        ) <= 0.30:
            score += self.timestamp_bonus

        avg_conf = (
            whisper["confidence"] +
            medasr["confidence"]
        ) / 2

        if avg_conf >= 0.90:
            score += self.confidence_bonus

        return score
    
    def _local_align(
        self,
        whisper_tokens,
        medasr_tokens
    ):

        n = len(whisper_tokens)
        m = len(medasr_tokens)

        dp = [
            [0]*(m+1)
            for _ in range(n+1)
        ]

        for i in range(1, n+1):
            dp[i][0] = dp[i-1][0] + self.gap_score

        for j in range(1, m+1):
            dp[0][j] = dp[0][j-1] + self.gap_score
            
        for i in range(1, n+1):

            wi = whisper_tokens[i-1]

            for j in range(1, m+1):

                mj = medasr_tokens[j-1]

                diagonal = (
                    dp[i-1][j-1] +
                    self.score(wi, mj)
                )

                up = (
                    dp[i-1][j] +
                    self.gap_score
                )

                left = (
                    dp[i][j-1] +
                    self.gap_score
                )

                dp[i][j] = max(
                    diagonal,
                    up,
                    left
                )
        aligned = []

        i = n
        j = m

        while i > 0 or j > 0:

            if (
                i > 0 and
                j > 0 and
                dp[i][j]
                ==
                dp[i-1][j-1]
                +
                self.score(
                    whisper_tokens[i-1],
                    medasr_tokens[j-1]
                )
            ):

                aligned.append({

                    "whisper": whisper_tokens[i-1],

                    "medasr": medasr_tokens[j-1]

                })

                i -= 1
                j -= 1

            elif (
                i > 0
                and
                dp[i][j]
                ==
                dp[i-1][j]
                +
                self.gap_score
            ):

                aligned.append({

                    "whisper": whisper_tokens[i-1],

                    "medasr": None

                })

                i -= 1

            else:

                aligned.append({

                    "whisper": None,

                    "medasr": medasr_tokens[j-1]

                })

                j -= 1

        aligned.reverse()

        return aligned
    
    def extract_anchors(
    self,
    whisper_tokens,
    medasr_tokens,
    matched_pairs
):
        """
        Convert TimestampMatcher output into anchors.

        Returns
        -------
        [
            {
                "whisper_idx": int,
                "medasr_idx": int,
                "whisper": token,
                "medasr": token
            }
        ]
        """

        anchors = []

        whisper_cursor = 0
        medasr_cursor = 0

        for pair in matched_pairs:

            if pair["whisper"] is None:
                continue

            if pair["medasr"] is None:
                continue

            w_idx = None
            for i in range(
                whisper_cursor,
                len(whisper_tokens)
            ):

                t = whisper_tokens[i]

                if (
                    t["token"] == pair["whisper"]["token"]
                    and
                    abs(
                        t["start"] -
                        pair["whisper"]["start"]
                    ) < 1e-6
                ):
                    w_idx = i
                    whisper_cursor = i + 1
                    break

            m_idx = None

            for j in range(
                medasr_cursor,
                len(medasr_tokens)
            ):

                t = medasr_tokens[j]

                if (
                    t["token"] == pair["medasr"]["token"]
                    and
                    abs(
                        t["start"] -
                        pair["medasr"]["start"]
                    ) < 1e-6
                ):
                    m_idx = j
                    medasr_cursor = j + 1
                    break

            if (
                w_idx is None
                or
                m_idx is None
            ):
                continue

            anchors.append({

                "whisper_idx": w_idx,
                "medasr_idx": m_idx,

                "whisper": pair["whisper"],
                "medasr": pair["medasr"]

            })

        return anchors
    
    def deduplicate_anchors(
        self,
        anchors
    ):

        seen = set()

        result = []

        for a in anchors:

            key = (
                a["whisper_idx"],
                a["medasr_idx"]
            )

            if key in seen:
                continue

            seen.add(key)

            result.append(a)

        return result
    
    def sort_anchors(
    self,
    anchors
):

        anchors.sort(
            key=lambda x: (
                x["whisper_idx"],
                x["medasr_idx"]
            )
        )

        return anchors
    
    def repair_anchors(
    self,
    anchors
):

        if not anchors:
            return []

        repaired = [anchors[0]]

        last_w = anchors[0]["whisper_idx"]
        last_m = anchors[0]["medasr_idx"]

        for anchor in anchors[1:]:

            if (
                anchor["whisper_idx"] > last_w
                and
                anchor["medasr_idx"] > last_m
            ):

                repaired.append(anchor)

                last_w = anchor["whisper_idx"]
                last_m = anchor["medasr_idx"]

        return repaired
    def build_anchors(self, matched_pairs):

        anchors = []

        for pair in matched_pairs:

            if pair["whisper"] is None:
                continue

            if pair["medasr"] is None:
                continue

            anchors.append({

                "whisper_idx": pair["whisper_idx"],

                "medasr_idx": pair["medasr_idx"],

                "whisper": pair["whisper"],

                "medasr": pair["medasr"]

            })

        anchors = self.deduplicate_anchors(anchors)
        anchors = self.sort_anchors(anchors)
        anchors = self.repair_anchors(anchors)

        return anchors
    def _window(self,
            whisper_tokens,
            medasr_tokens,
            w_start,
            w_end,
            m_start,
            m_end):

        return (

            whisper_tokens[w_start:w_end],

            medasr_tokens[m_start:m_end]

        )
        
    def build_windows(
    self,
    whisper_tokens,
    medasr_tokens,
    anchors
):

        """
        Creates independent alignment windows.

        Window

        previous anchor
            ↓

        whisper[prev_w+1 : curr_w]

        medasr[prev_m+1 : curr_m]

        """

        windows = []

        prev_w = -1
        prev_m = -1
        for anchor in anchors:

            curr_w = anchor["whisper_idx"]

            curr_m = anchor["medasr_idx"]

            w_tokens, m_tokens = self._window(

                whisper_tokens,

                medasr_tokens,

                prev_w,

                curr_w,

                prev_m,

                curr_m

            )

            windows.append({

                "type": "window",

                "whisper": w_tokens,

                "medasr": m_tokens

            })

            windows.append({

                "type": "anchor",

                "pair": {

                    "whisper": anchor["whisper"],

                    "medasr": anchor["medasr"]

                }

            })

            prev_w = curr_w + 1

            prev_m = curr_m + 1
        tail_w = whisper_tokens[prev_w + 1 : ]


        tail_m = medasr_tokens[prev_m + 1 : ]

        windows.append({

            "type": "window",

            "whisper": tail_w,

            "medasr": tail_m

        })

        return windows
    
    def align(
    self,
    whisper_tokens,
    medasr_tokens,
    matched_pairs=None
):
        """
        Anchored Needleman-Wunsch alignment.

        Steps
        -----
        1. Build timestamp anchors.
        2. Segment transcript into windows.
        3. Align only window regions.
        4. Copy anchors directly.
        5. Merge everything.
        """

        # Fallback to global NW if no anchors exist
        if matched_pairs is None:
            return self._local_align(
                whisper_tokens,
                medasr_tokens
            )

        anchors = self.build_anchors(
            matched_pairs
        )

        if len(anchors) == 0:
            return self._local_align(
                whisper_tokens,
                medasr_tokens
            )

        result = []

        prev_w = -1
        prev_m = -1

        for anchor in anchors:

            curr_w = anchor["whisper_idx"]
            curr_m = anchor["medasr_idx"]

            whisper_window = whisper_tokens[
                prev_w + 1 : curr_w
            ]

            medasr_window = medasr_tokens[
                prev_m + 1 : curr_m
            ]

            if whisper_window or medasr_window:

                result.extend(

                    self._local_align(
                        whisper_window,
                        medasr_window
                    )

                )

            result.append({

                "whisper": anchor["whisper"],

                "medasr": anchor["medasr"]

            })

            prev_w = curr_w
            prev_m = curr_m

        whisper_tail = whisper_tokens[
            prev_w + 1 :
        ]

        medasr_tail = medasr_tokens[
            prev_m + 1 :
        ]

        if whisper_tail or medasr_tail:

            result.extend(

                self._local_align(
                    whisper_tail,
                    medasr_tail
                )

            )

        return result
    
    
    
    
    