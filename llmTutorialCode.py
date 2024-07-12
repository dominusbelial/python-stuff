#Step2: Create tokenizers

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

path_en = [str(file) for file in Path('./dataset-en').glob("**/*.txt")]
path_my = [str(file) for file in Path('./dataset-my').glob("**/*.txt")]

# Create Source Tokenizer - English
tokenizer_en = Tokenizer(BPE(unk_token="[UNK]"))
trainer_en = BpeTrainer(min_frequency=2, special_tokens=["[PAD]","[UNK]","[CLS]", "[SEP]", "[MASK]"])
# We’ll also need to add a pre-tokenizer to split our input into words as without a pre-tokenizer, we might get tokens that overlap several words: for instance we could get a "there is" token since those two words often appear next to each other.
# Using a pre-tokenizer will ensure no token is bigger than a word returned by the pre-tokenizer.
tokenizer_en.pre_tokenizer = Whitespace()
tokenizer_en.train(files=path_en, trainer=trainer_en)
tokenizer_en.save("./tokenizer_en/tokenizer_en.json")

# Create Target Tokenizer - Malay
tokenizer_my = Tokenizer(BPE(unk_token="[UNK]"))
trainer_my = BpeTrainer(min_frequency=2, special_tokens=["[PAD]","[UNK]","[CLS]", "[SEP]", "[MASK]"])
tokenizer_my.pre_tokenizer = Whitespace()
tokenizer_my.train(files=path_my, trainer=trainer_my)
tokenizer_my.save("./tokenizer_my/tokenizer_my.json")

tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer_en.json")
tokenizer_my = Tokenizer.from_file("./tokenizer_my/tokenizer_my.json")

source_vocab_size = tokenizer_en.get_vocab_size()
target_vocab_size = tokenizer_my.get_vocab_size()

CLS_ID = torch.tensor([tokenizer_my.token_to_id("[CLS]")], dtype=torch.int64).to(device)
SEP_ID = torch.tensor([tokenizer_my.token_to_id("[SEP]")], dtype=torch.int64).to(device)
PAD_ID = torch.tensor([tokenizer_my.token_to_id("[PAD]")], dtype=torch.int64).to(device)