from wordfreq import zipf_frequency

class TokenReconstructor:

    """

    Token Reconstruction
 
    Goal

    ----

    Merge only genuine ASR sub-word fragments.
 
    Examples

    --------

    met + in      -> metin (rare)

    a + fib       -> afib
 
    Do NOT merge:
 
    taking + met

    chest + pain

    have + fever

    """
 
    COMMON_WORDS = {

        "a", "i", "in", "on", "at", "to", "of", "is", "it",

        "be", "do", "so", "no", "up", "we", "he", "she", "if",

        "and", "the", "you", "are", "was", "for", "but", "not",

        "can", "has", "had", "his", "her", "its", "our", "out",

        "all", "any", "did", "get", "got", "how", "let", "may",

        "now", "off", "old", "one", "own", "see", "too", "use",

        "yes", "yet", "or", "an",

    }
 
    # ONLY genuine fragments

    MAX_FRAGMENT_LEN = 3
 
    # fragments must be extremely close

    MAX_GAP = 0.08
 
    MAX_MERGE = 5
 
    def __init__(self):

        pass
 
    # ---------------------------------------------------------
 
    def _is_seed(self, token_str):
 
        t = token_str.lower()
 
        if t.isdigit():

            return False
 
        if zipf_frequency(t, "en") > 3:
            return False
 
        return len(t) <= self.MAX_FRAGMENT_LEN
 
    # ---------------------------------------------------------
 
    def _is_continuation(self, token_str):
 
        t = token_str.lower()
 
        if t.isdigit():

            return False
 
        return len(t) <= self.MAX_FRAGMENT_LEN
 
    # ---------------------------------------------------------
 
    def reconstruct(self, segments):
 
        reconstructed = []
 
        for segment in segments:
 
            words = segment["words"]
 
            merged = []
 
            i = 0
 
            while i < len(words):
 
                current = words[i]
 
                candidate = current["token"]
 
                start = current["start"]
 
                end = current["end"]
 
                conf_sum = current["confidence"]
 
                conf_count = 1
 
                consumed = 1
 
                if self._is_seed(candidate):
 
                    merged_any = False
 
                    for j in range(1, self.MAX_MERGE):
 
                        if i + j >= len(words):

                            break
 
                        nxt = words[i + j]
 
                        gap = nxt["start"] - end
 
                        if gap > self.MAX_GAP:

                            break
 
                        if not self._is_continuation(

                            nxt["token"]

                        ):

                            break
 
                        candidate += nxt["token"]
 
                        end = nxt["end"]
 
                        conf_sum += nxt["confidence"]
 
                        conf_count += 1
 
                        consumed += 1
 
                        merged_any = True
 
                    if not merged_any:
 
                        candidate = current["token"]
 
                        consumed = 1
 
                confidence = round(

                    conf_sum / conf_count,

                    2

                )
 
                merged.append(

                    {

                        "token": candidate,

                        "start": start,

                        "end": end,

                        "confidence": confidence,

                    }

                )
 
                i += consumed
 
            reconstructed.append(

                {

                    "segment_id": segment["segment_id"],

                    "speaker": segment["speaker"],

                    "words": merged,

                }

            )
 
        return reconstructed
 
    # ---------------------------------------------------------
 
    def flatten(self, segments):
 
        tokens = []
 
        for segment in segments:
 
            speaker = segment["speaker"]
 
            seg = segment["segment_id"]
 
            for word in segment["words"]:
 
                token = word.copy()
 
                token["speaker"] = speaker
 
                token["segment_id"] = seg
 
                tokens.append(token)
 
        return tokens
 
    # ---------------------------------------------------------
 
    def display(self, segments, title):
 
        print()

        print("=" * 80)

        print("TOKEN RECONSTRUCTION")

        print("=" * 80)

        print()

        print(title)

        print()
 
        total = sum(

            len(seg["words"])

            for seg in segments

        )
 
        print("Total Tokens :", total)

        print()
 
        print("Sample Tokens")
 
        shown = 0
 
        for segment in segments:
 
            for word in segment["words"]:
 
                print(

                    f'{word["token"]:<20}'

                    f'{word["start"]:.2f}-{word["end"]:.2f}   '

                    f'conf={word["confidence"]:.2f}   '

                    f'{segment["speaker"]}'

                )
 
                shown += 1
 
                if shown >= 10:

                    break
 
            if shown >= 10:

                break
 
        print()

        print("=" * 80)
 