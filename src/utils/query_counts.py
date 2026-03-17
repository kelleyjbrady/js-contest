            SELECT 
              "source",
              --eval_status,
                count(prompt_id)
            FROM prompts 
            --WHERE eval_status = 'graded' 
              --AND source IS NOT NULL
              --and source = 'augmented_suspicious'
            group by "source"--, eval_status