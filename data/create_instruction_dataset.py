from collections import defaultdict
import json

def create_language_pairs():
    pairs = defaultdict(lambda: {"en": '', 'gr': '', 'he': ''})
    with open('data/berean_bible.txt') as en_f, open('data/tr.txt') as gr_f, open('data/he_modern.txt') as he_f:
        en_lines = en_f.readlines()
        gr_lines = gr_f.readlines()
        he_lines = he_f.readlines()
    
    for en_line, he_line in zip(en_lines,he_lines):
        en_line = en_line.strip()
        if not en_line and not he_line:
            continue

        # First try tab-separated format (English Genesis)
        if '\t' in en_line:
            parts = en_line.split('\t', 1)
            if len(parts) == 2:
                ref, text = parts
                pairs[ref.strip()]['en'] = text.strip()
    
        parts = he_line.split(' ')
                
        # Look for the chapter:verse pattern to determine where reference ends
        ref_end_idx = -1
        for i, part in enumerate(parts):
            if ':' in part:
                ref_end_idx = i
                break
        
        if ref_end_idx >= 0:            
            # The rest is the text
            text = ' '.join(parts[ref_end_idx+1:])
            
            # Clean up any special markers like ¶
            text = text.replace('¶', '').strip()
            
            pairs[ref.strip()]['he'] = text
        
    for gr_line in gr_lines:
        parts = gr_line.split(' ')
            
        # Look for the chapter:verse pattern to determine where reference ends
        ref_end_idx = -1
        for i, part in enumerate(parts):
            if ':' in part:
                ref_end_idx = i
                break
        
        if ref_end_idx >= 0:
            # Combine book name (might have spaces) with chapter:verse
            ref_parts = parts[:ref_end_idx+1]
            reference = ' '.join(ref_parts).strip()
            
            # The rest is the text
            text = ' '.join(parts[ref_end_idx+1:])
            
            # Clean up any special markers like ¶
            text = text.replace('¶', '').strip()
            
            pairs[reference]['gr'] = text
    
    # Filter out entries where any language is missing
    complete_pairs = {
        ref: langs
        for ref, langs in pairs.items()
        if langs['en'] and langs['he']
    }
    return complete_pairs

if __name__ == '__main__':
    pairs = create_language_pairs()
    formatted = []
    system_prompt = "You are a helpful and respectful assistant with deep knowledge of the Bible, Christian theology, history, and traditions. You answer questions clearly and compassionately, citing Scripture when appropriate and remaining sensitive to different Christian perspectives. When possible, provide references (e.g., book, chapter, and verse) to support your responses. If a question is theological or interpretive, acknowledge differing views graciously and stay grounded in biblical context. Your goal is to inform, guide, and encourage users with wisdom and humility."
    for verse, language_text in pairs.items():
        if language_text['gr']:
            formatted.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Translate {verse}: {language_text["en"]} into greek'},
                    {"role": "assistant", "content": f'{language_text["gr"]}'}  # Placeholder for the actual translation
                ]
            })
            formatted.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Translate {verse}: {language_text["gr"]} into english'},
                    {"role": "assistant", "content": f'{language_text["en"]}'}  # Placeholder for the actual translation
                ]
            })
        formatted.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Translate {verse}: {language_text["en"]} into modern hebrew'},
                    {"role": "assistant", "content": f'{language_text["he"]}'}  # Placeholder for the actual translation
                ]
            })
        formatted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Translate {verse}: {language_text["he"]} into english'},
                {"role": "assistant", "content": f'{language_text["en"]}'}  # Placeholder for the actual translation
            ]
        })

    with open("instruction_tuned_translation.jsonl", "w", encoding="utf-8") as f:
        for item in formatted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")