class TimestampMatcher:
    """
    Timestamp Based Candidate Matching

    Matches Whisper and MedASR tokens based on
    temporal proximity before Needleman alignment.

    Output
    ------
    [
        {
            "whisper": {...},
            "medasr": {...}
        }
    ]
    """

    def __init__(self, tolerance=0.50):

        # allowed timestamp difference (seconds)

        self.tolerance = tolerance

    # ---------------------------------------------------------

    def match(self, whisper_tokens, medasr_tokens):

        matched = []

        used = set()

        for w_idx, w in enumerate(whisper_tokens):

            best = None
            best_diff = 999

            for idx, m in enumerate(medasr_tokens):

                if idx in used:
                    continue

                # speaker mismatch

                if w["speaker"] != m["speaker"]:
                    continue

                diff = abs(

                    w["start"] -

                    m["start"]

                )

                if diff <= self.tolerance:

                    if diff < best_diff:

                        best = idx
                        best_diff = diff

            if best is not None:

                matched.append(

                    {
                        "whisper_idx": w_idx,
                        "medasr_idx": best,

                        "whisper": w,

                        "medasr": medasr_tokens[best]

                    }

                )

                used.add(best)

            else:

                matched.append(

                    {
                        "whisper_idx": w_idx,
                        "medasr_idx": None,

                        "whisper": w,

                        "medasr": None

                    }

                )

        # remaining MedASR words

        for idx, token in enumerate(medasr_tokens):

            if idx not in used:

                matched.append(

                    {
                        "whisper_idx": None,
                        "medasr_idx": idx,

                        "whisper": None,

                        "medasr": token

                    }

                )

        # sort by timestamp

        matched.sort(

            key=lambda x:

            (

                x["whisper"]["start"]

                if x["whisper"]

                else x["medasr"]["start"]

            )

        )

        return matched

    # ---------------------------------------------------------

    def display(self, matched):

        print()

        print("=" * 80)

        print("TIMESTAMP MATCHER")

        print("=" * 80)

        print()

        for pair in matched[:20]:

            w = "-"

            m = "-"

            if pair["whisper"]:

                w = pair["whisper"]["token"]

            if pair["medasr"]:

                m = pair["medasr"]["token"]

            print(

                f"{w:<25}"

                f"{m:<25}"

            )

        print()

        print(

            "Candidate Pairs :",

            len(matched)

        )

        print()

        print("=" * 80)