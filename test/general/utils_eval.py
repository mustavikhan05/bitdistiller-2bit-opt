import transformers
import torch
from lm_eval.base import BaseLM

class LMEvalAdaptor(BaseLM):

    def __init__(self, model_name, model, tokenizer, batch_size=1, max_length=-1):
        super().__init__()

        assert isinstance(batch_size, int)

        self.model_name = model_name
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self._batch_size = batch_size

        self._max_length = max_length

        # Set tokenizer to avoid truncation
        self.tokenizer.model_max_length = 1e30  # Prevent tokenizer from truncating
        # Optionally, keep track of actual max length
        self.tokenizer.init_kwargs['model_max_length'] = self.max_length

    @property
    def eot_token_id(self):
        # Use End-of-Text token ID
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length != -1:
            return self._max_length
        if hasattr(self.model.config, 'n_ctx'):
            return self.model.config.n_ctx
        elif hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        elif 'bloom' in self.model_name.lower():
            return 2048
        elif 'llama' in self.model_name.lower():
            return 2048
        elif 'mpt' in self.model_name.lower():
            return 2048
        elif 'falcon' in self.model_name.lower():
            return 2048
        elif 'opt' in self.model_name.lower():
            return self.model.config.max_position_embeddings  # For OPT models
        else:
            print(self.model.config)
            raise NotImplementedError

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def tok_encode(self, string: str):
        return self.tokenizer.encode(
            string,
            add_special_tokens=False
        )

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        The size of sequence may vary from call to call.

        Returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model.
        """
        with torch.no_grad():
            inps = inps.to(self.device)
            # Truncate input to max_length from the left
            if inps.size(1) > self.max_length:
                inps = inps[:, -self.max_length:]

            if isinstance(self.model, transformers.T5ForConditionalGeneration):
                dec_inps = torch.cat(
                    [
                        torch.tensor(
                            [self.model.config.decoder_start_token_id],
                            device=self.device,
                        )
                        .repeat(len(inps), 1),
                        inps,
                    ],
                    dim=1,
                )

                kwargs = {"decoder_input_ids": dec_inps}
            else:
                kwargs = {}

            outputs = self.model(inps, **kwargs)

            # Access logits based on model output structure
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]

            if "opt" in self.model_name.lower():
                # OPT models may have extra tokens; adjust logits accordingly
                return logits[:, :, :self.tokenizer.vocab_size]
            else:
                return logits

    def _model_generate(self, context, max_length, eos_token_id):
        context = context.to(self.device)
        # Truncate context to max_length from the left
        if context.size(1) > self.max_length:
            context = context[:, -self.max_length:]

        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )