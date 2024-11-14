#ifndef __MY_REGISTER__
#define __MY_REGISTER__

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "assert.h"
#include "../abstract_hardware_model.h"

class register_base 
{
public:
    register_base () {}
    register_base (
        unsigned bank_num = 32, 
        unsigned bank_num_per_scheduler = 8,
        unsigned entry_num_per_bank = 64
        ) : m_register( bank_num, std::vector < register_entry > ( entry_num_per_bank ) ), 
            m_entry_num_per_bank( entry_num_per_bank ),
            m_bank_num ( bank_num ),
            m_bank_num_per_sheduler ( bank_num_per_scheduler ) {}
    ~register_base () {}

    void init ();

    bool set_register_entry (
        int bank_id,
        int register_id,
        int warp_id
    );
    int register2bank (
        int register_id,
        int warp_id,
        int bank_warp_shift,
        bool sub_core_model,
        int scheduler_id
        );
    void print_reg_entry_count ();
public:
    enum register_state 
    {
        INACTIVE = 0,
        REG = 1,      
    };
    struct register_entry
    {
        enum register_state m_register_state;
        int m_warp_id;
        int m_thread_id;
        int m_regsiter_id;
    };
public:
    bool set_register_entry (
        int bank_id,
        int register_id,
        int warp_id,
        register_state state
    );
protected:

    std::vector < std::vector <register_entry> > m_register; 
    unsigned m_entry_num_per_bank;
    unsigned m_bank_num;
    unsigned m_bank_num_per_sheduler;

private:
    unsigned m_reg_entry_count;
};


inline 
int register_base::register2bank (        
        int register_id,
        int warp_id,
        int bank_warp_shift,
        bool sub_core_model,
        int scheduler_id ) 
{
    int bank_id = register_id;
    if ( bank_warp_shift ) bank_id += warp_id;
    if ( sub_core_model ) 
    {
        // sub-core model
        bank_id = ( bank_id % m_bank_num_per_sheduler ) + 
                  ( scheduler_id * m_bank_num_per_sheduler );
        assert ( bank_id < (int)m_bank_num );
        return bank_id;
    }
    else 
    {   
        // core model
        return bank_id % m_bank_num;
    }
}

inline 
void register_base::init () 
{
    m_reg_entry_count = 0;
  
    for ( unsigned i = 0; i < m_bank_num; i++ ) 
    {
        for (unsigned j = 0; j < m_entry_num_per_bank; j++ ) 
        {
            m_register[i][j].m_register_state = INACTIVE;
            m_register[i][j].m_warp_id = -1;
            m_register[i][j].m_thread_id = -1;
            m_register[i][j].m_regsiter_id = -1;
        }
    }
}

inline 
bool register_base::set_register_entry (
    int bank_id,
    int register_id,
    int warp_id )
{
   
    return false;
}

inline 
bool register_base::set_register_entry (
    int bank_id,
    int register_id,
    int warp_id,
    register_state state )
{
    
    for ( auto& elem : m_register[bank_id] )
    {   

        if ( elem.m_regsiter_id == register_id && elem.m_warp_id == warp_id )
        {
            elem.m_register_state = state;
           
            return true;  
        }
    }
    for ( auto& elem : m_register[bank_id] )
    {   
       
        if ( elem.m_register_state == INACTIVE && elem.m_regsiter_id == -1 && 
             elem.m_warp_id == -1 )
        {
            elem.m_regsiter_id = register_id;
            elem.m_warp_id = warp_id;
            elem.m_register_state = state;
            
            return true;  
        }
    }
    return false;  
}

inline
void register_base::print_reg_entry_count ()
{
    printf("m_reg_entry_cout=%u\n", m_reg_entry_count);
}


#endif