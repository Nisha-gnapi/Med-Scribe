import re


class LightNormalizer:

    """
    Light normalization for medical transcripts.

    Operations
    ----------
    ✓ lowercase
    ✓ remove punctuation
    ✓ preserve numbers
    ✓ preserve timestamps
    ✓ preserve confidence
    """

    def __init__(self):

        self.pattern = re.compile(r"[^\w\s\-]")

    # -----------------------------------------------------

    def normalize_token(self, token):

        token = token.lower()

        token = self.pattern.sub("", token)

        token = token.strip()

        return token

    # -----------------------------------------------------

    def normalize(self, segments):

        normalized = []

        for segment in segments:

            new_segment = {

                "segment_id": segment["segment_id"],
                "speaker": segment["speaker"],
                "words": []

            }

            for word in segment["words"]:

                new_word = word.copy()

                new_word["token"] = self.normalize_token(

                    word["token"]

                )

                if new_word["token"] != "":

                    new_segment["words"].append(

                        new_word

                    )

            normalized.append(

                new_segment

            )

        return normalized

    # -----------------------------------------------------

    def display(self, segments, title):

        print()

        print("=" * 80)

        print("LIGHT NORMALIZER")

        print("=" * 80)

        print()

        print(title)

        print()

        total = 0

        sample = []

        for segment in segments:

            total += len(segment["words"])

            for word in segment["words"]:

                if len(sample) < 10:

                    sample.append(

                        word["token"]

                    )

        print("Words Processed :", total)

        print()

        print("Sample Output")

        print(" ".join(sample))

        print()

        print("=" * 80)